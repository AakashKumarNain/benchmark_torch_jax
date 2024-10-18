import math
import inspect
from dataclasses import dataclass

import tiktoken

import jax
import math
import jax.numpy as jnp
from jax import tree_util as jtu

import equinox as eqx
from equinox._misc import default_floating_dtype


######################## Equinox model utils ################################


def is_layer(x):
    """Check if the current pytree is an instance of any Equinox layer."""
    return isinstance(x, (eqx.nn.Linear, eqx.nn.Embedding, eqx.nn.LayerNorm))


def is_leaf(x):
    """Check if the current node is a leaf"""
    return x is None


def set_mask(x):
    """Sets the mask for certain parameters.

    There are scenarios where you want to filter out the parameters of the
    model for applying some specialized op. For example, in this case we
    are filtering our pytree and masking certain parameters to avoid applying
    `weight_decay` to these parameters. These parameters are:

    1. Linear layer -> Weight decay is only applied to the weights and not the bias
    2. Embedding -> Weight decay applied to the weights
    3. Any other layer e.g. LayerNorm -> No weight decay is applied
    """

    if isinstance(x, eqx.nn.Linear):
        # Decay has to be applied only on the weights, and not the biases
        mask = jtu.tree_map(lambda _: True, x)
        mask = eqx.tree_at(lambda m: m.bias, mask, False, is_leaf=is_leaf)
        return mask
    elif isinstance(x, eqx.nn.Embedding):
        return jtu.tree_map(lambda _: True, x)
    else:
        return jtu.tree_map(lambda _: False, x)


def count_params(model):
    """Count the parameters in an Equinox model"""
    return sum(x.size for x in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


def get_weight_and_bias(module):
    """Return the weight and bias(if present) in a layer/module."""
    if hasattr(module, "bias") and module.bias is not None:
        return module.weight, module.bias
    return module.weight


def set_weight_and_bias(weight, bias, key, mean=0.0, std=0.02):
    """Set the weight and bias of a layer.

    The weights are drawn from a normal distribution with a
    mean value of 0.0 and a given std. The bias (if present)
    is set to zeros.

    Args:
        weight: Current value of the weight matrix
        bias: Current value of the bias vector
        key: Pseudo random key for sampling data
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
    Returns:
        New Weight and bias (if not None) values
    """

    init = jax.nn.initializers.normal(stddev=std)
    weight = init(key=key, shape=weight.shape).astype(weight.dtype)
    if bias is not None:
        bias = jnp.zeros_like(bias, dtype=weight.dtype)
        return weight, bias
    return weight


###############################################################


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(self, config, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype() if dtype is None else dtype
        key1, key2 = jax.random.split(key, 2)
        std = 0.02

        self.fc1 = eqx.nn.Linear(
            config.embed_dim, config.embed_dim * 4, key=key1, dtype=dtype
        )
        self.proj = eqx.nn.Linear(
            config.embed_dim * 4, config.embed_dim, key=key2, dtype=dtype
        )

        # Set the weights and bias of the linear layer as per the paper
        self.fc1 = eqx.tree_at(
            get_weight_and_bias,
            self.fc1,
            set_weight_and_bias(self.fc1.weight, self.fc1.bias, key1, std=std),
        )
        # Set the weights and bias of the projection layer as per the paper
        self.proj = eqx.tree_at(
            get_weight_and_bias,
            self.proj,
            set_weight_and_bias(
                self.proj.weight,
                self.proj.bias,
                key2,
                std * (2 * config.num_layers) ** -0.5,
            ),
        )

    def __call__(self, x):
        x = eqx.filter_vmap(self.fc1)(x)
        x = jax.nn.gelu(x.astype(jnp.float32))
        x = eqx.filter_vmap(self.proj)(x.astype(jnp.bfloat16))
        return x


class CausalSelfAttention(eqx.Module):
    num_heads: int
    num_layers: int
    wqkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    scale: float

    def __init__(self, config, key, dtype=jnp.bfloat16):
        assert config.embed_dim % config.num_heads == 0
        dtype = default_floating_dtype() if dtype is None else dtype
        key1, key2 = jax.random.split(key, 2)

        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.scale = 1.0 / math.sqrt(config.embed_dim)

        self.wqkv = eqx.nn.Linear(
            config.embed_dim, 3 * config.embed_dim, key=key1, dtype=dtype
        )
        self.proj = eqx.nn.Linear(
            config.embed_dim, config.embed_dim, key=key2, dtype=dtype
        )

        self.wqkv = eqx.tree_at(
            get_weight_and_bias,
            self.wqkv,
            set_weight_and_bias(
                self.wqkv.weight,
                self.wqkv.bias,
                key1,
                std=0.02,
            ),
        )
        self.proj = eqx.tree_at(
            get_weight_and_bias,
            self.proj,
            set_weight_and_bias(
                self.proj.weight,
                self.proj.bias,
                key2,
                std=0.02 * (2 * config.num_layers) ** -0.5,
            ),
        )

    def __call__(self, x, mask=None):
        # x is of shape [seqlen, embed_dim]
        # batch size will be handled by vmap
        T, C = x.shape

        # 1. Calculate qkv
        qkv = eqx.filter_vmap(self.wqkv)(x)

        # 2. Split qkv into three vectors of equal depth
        q, k, v = jnp.split(qkv, 3, axis=1)

        # 3. Reshape q, k,v to move the heads to the batch dimension
        # so that we can calculate the attention for all heads in one go
        q = jnp.reshape(q, (T, self.num_heads, C // self.num_heads))
        k = jnp.reshape(k, (T, self.num_heads, C // self.num_heads))
        v = jnp.reshape(v, (T, self.num_heads, C // self.num_heads))

        # 4. Compute attention
        attn = jax.nn.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = jnp.reshape(attn.astype(jnp.bfloat16), (T, -1))

        # 5. Projection
        out = eqx.filter_vmap(self.proj)(attn)
        return out


class TransformerBlock(eqx.Module):
    norm_1: eqx.nn.LayerNorm
    norm_2: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config, key, dtype=jnp.bfloat16):
        key1, key2 = jax.random.split(key, 2)
        self.norm_1 = eqx.nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config, key=key1, dtype=dtype)
        self.norm_2 = eqx.nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config, key=key2, dtype=dtype)

    def __call__(self, x, mask=None):
        x_dtype = x.dtype
        x = eqx.filter_vmap(self.norm_1).astype(x_dtype)
        x = x + self.attn(x, mask=mask)
        x = eqx.filter_vmap(self.norm_2)(x).astype(x_dtype)
        x = x + self.mlp(x)
        return x


class GPT(eqx.Module):
    block_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    tok_embed_and_head: eqx.nn.Shared
    pos_embed: eqx.nn.Embedding
    tf_blocks: TransformerBlock
    norm: eqx.nn.LayerNorm

    def __init__(self, config, key, dtype=jnp.bfloat16):
        self.block_size = config.block_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.vocab_size = config.vocab_size

        keys = jax.random.split(key, config.num_layers + 3)
        key1, key2, key3, tf_keys = keys[0], keys[1], keys[2], keys[3:]

        self.norm = eqx.nn.LayerNorm(config.embed_dim)

        make_layers = lambda k: TransformerBlock(config, key=k, dtype=dtype)
        self.tf_blocks = eqx.filter_vmap(make_layers)(tf_keys)
        del make_layers

        self.pos_embed = eqx.nn.Embedding(
            config.block_size, config.embed_dim, key=key1, dtype=dtype
        )
        self.pos_embed = eqx.tree_at(
            get_weight_and_bias,
            self.pos_embed,
            set_weight_and_bias(self.pos_embed.weight, None, key1),
        )

        tok_embed = eqx.nn.Embedding(
            config.vocab_size, config.embed_dim, key=key2, dtype=dtype
        )
        tok_embed = eqx.tree_at(
            get_weight_and_bias,
            tok_embed,
            set_weight_and_bias(tok_embed.weight, None, key2),
        )

        lm_head = eqx.nn.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, key=key3, dtype=dtype
        )
        dst = lambda embed_and_linear: embed_and_linear[1].weight
        src = lambda embed_and_linear: embed_and_linear[0].weight
        self.tok_embed_and_head = eqx.nn.Shared((tok_embed, lm_head), dst, src)

    def __call__(self, idx, mask=None):
        tok_embed, lm_head = self.tok_embed_and_head()
        seqlen = idx.shape[-1]
        pos = jnp.arange(0, seqlen, dtype=jnp.int32)

        # idx is of shape (seqlen,)
        pos_embed = eqx.filter_vmap(self.pos_embed)(pos)
        tok_embed = eqx.filter_vmap(tok_embed)(idx)

        # 2. Add position to token embeddings
        x = pos_embed + tok_embed

        # 3. Partition the TransformerLayers into static and dynamic parts
        # and pass the previous output through transformer blocks
        dynamic_layers, static_layers = eqx.partition(self.tf_blocks, eqx.is_array)
        layer_idx = 0

        def f(_x, _dynamic_l):
            layer = eqx.combine(_dynamic_l, static_layers)
            x, layer_idx = _x
            x = layer(x)
            return (x, layer_idx + 1), None

        (x, layer_idx), _ = jax.lax.scan(f, (x, layer_idx), dynamic_layers)

        # 4. Final pre-layer norm
        x = eqx.filter_vmap(self.norm)(x).astype(jnp.bfloat16)

        # 5. Classification head
        logits = eqx.filter_vmap(lm_head)(x)
        return logits


###############################################################


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = jnp.array(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        batch_tokens = self.tokens[
            self.current_position : self.current_position + B * T + 1
        ]
        x = jnp.reshape(batch_tokens[:-1], (B, T))
        y = jnp.reshape(batch_tokens[1:], (B, T))
        self.current_position += B * T

        # Check if we already processed the last batch
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

    def reset(self):
        self.current_position = 0


###############################################################
