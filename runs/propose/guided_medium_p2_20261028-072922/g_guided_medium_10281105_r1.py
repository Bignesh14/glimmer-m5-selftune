"""fuse_qkv_mlp — fuse QKV and MLP projections to reduce kernel launches."""

NAME = "g_guided_medium_10281105_r1"
EXACT = True
DESCRIPTION = "Fuse QKV and MLP projections with in-place weight freeing to stay under 25GB."

def apply():
    import mlx.nn as nn
    from mlx_lm.models.muse_glimmer import Attention, MLP
    from mlx_lm.models.base import scaled_dot_product_attention

    _orig_attn_call = Attention.__call__
    _orig_mlp_call = MLP.__call__

    def fused_attention_call(self, x, mask=None, cache=None):
        if not hasattr(self, "qkv_fused_weight"):
            return _orig_attn_call(self, x, mask, cache)

        B, L, _ = x.shape
        out = __import__("mlx.core", fromlist=["quantized_matmul"]).quantized_matmul(
            x,
            self.qkv_fused_weight,
            scales=self.qkv_fused_scales,
            biases=self.qkv_fused_biases,
            transpose=True,
            group_size=self.qkv_group_size,
            bits=self.qkv_bits,
            mode=self.qkv_mode,
        )
        s0, s1 = self.qkv_splits
        q = out[..., :s0]
        k = out[..., s0:s0 + s1]
        v = out[..., s0 + s1:]

        q = q.reshape(B, L, self.n_heads, self.head_dim)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim)

        q = __import__("mlx.core", fromlist=["fast"]).fast.rms_norm(q, None, self.eps) * self.qk_scale_factor
        k = __import__("mlx.core", fromlist=["fast"]).fast.rms_norm(k, None, self.eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out_attn = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out_attn = out_attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out_attn = out_attn * __import__("mlx.core", fromlist=["sigmoid"]).sigmoid(self.gate_proj(x))
        return self.o_proj(out_attn)

    def fused_mlp_call(self, x):
        if not hasattr(self, "gate_up_fused_weight"):
            return _orig_mlp_call(self, x)

        out = __import__("mlx.core", fromlist=["quantized_matmul"]).quantized_matmul(
            x,
            self.gate_up_fused_weight,
            scales=self.gate_up_fused_scales,
            biases=self.gate_up_fused_biases,
            transpose=True,
            group_size=self.gate_up_group_size,
            bits=self.gate_up_bits,
            mode=self.gate_up_mode,
        )
        inter = self.mlp_intermediate_size
        gate = out[..., :inter]
        up = out[..., inter:]
        return self.down_proj(nn.silu(gate) * up)

    Attention.__call__ = fused_attention_call
    MLP.__call__ = fused_mlp_call


def post_load(model):
    import mlx.core as mx

    for layer in model.model.layers:
        attn = layer.self_attn

        # save quantization params before freeing
        gs = attn.q_proj.group_size
        bits = attn.q_proj.bits
        mode = attn.q_proj.mode

        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        v_w = attn.v_proj.weight

        attn.qkv_fused_weight = mx.concatenate([q_w, k_w, v_w], axis=0)
        attn.qkv_fused_scales = mx.concatenate(
            [attn.q_proj.scales, attn.k_proj.scales, attn.v_proj.scales], axis=0
        )
        attn.qkv_fused_biases = mx.concatenate(
            [attn.q_proj.biases, attn.k_proj.biases, attn.v_proj.biases], axis=0
        )
        attn.qkv_splits = (q_w.shape[0], k_w.shape[0])
        attn.qkv_group_size = gs
        attn.qkv_bits = bits
        attn.qkv_mode = mode

        # free original weights to keep peak memory <25GB
        attn.q_proj.weight = None
        attn.q_proj.scales = None
        attn.q_proj.biases = None
        attn.k_proj.weight = None
        attn.k_proj.scales = None
        attn.k_proj.biases = None
        attn.v_proj.weight = None
        attn.v_proj.scales = None
        attn.v_proj.biases = None

        mlp = layer.mlp
        gs_m = mlp.gate_proj.group_size
        bits_m = mlp.gate_proj.bits
        mode_m = mlp.gate_proj.mode

        g_w = mlp.gate_proj.weight
        u_w = mlp.up_proj.weight

        mlp.gate_up_fused_weight = mx.concatenate([g_w, u_w], axis=0)
        mlp.gate_up_fused_scales = mx.concatenate(
            [mlp.gate_proj.scales, mlp.up_proj.scales], axis=0
        )
        mlp.gate_up_fused_biases = mx.concatenate(
            [mlp.gate_proj.biases, mlp.up_proj.biases], axis=0
        )
        mlp.mlp_intermediate_size = g_w.shape[0]
        mlp.gate_up_group_size = gs_m
        mlp.gate_up_bits = bits_m
        mlp.gate_up_mode = mode_m

        # free original weights
        mlp.gate_proj.weight = None
        mlp.gate_proj.scales = None
        mlp.gate_proj.biases = None
        mlp.up_proj.weight = None
        mlp.up_proj.scales = None
        mlp.up_proj.biases = None
