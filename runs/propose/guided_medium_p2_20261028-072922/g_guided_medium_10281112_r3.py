"""fuse_mlp_only — fuse MLP gate/up projections to reduce kernel launches."""

NAME = "g_guided_medium_10281112_r3"
EXACT = True
DESCRIPTION = "Fuse only MLP gate/up projections with in-place weight freeing."

def apply():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.muse_glimmer import MLP

    _orig_mlp_call = MLP.__call__

    def fused_mlp_call(self, x):
        if not hasattr(self, "gate_up_fused_weight"):
            return _orig_mlp_call(self, x)

        out = mx.quantized_matmul(
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

    MLP.__call__ = fused_mlp_call


def post_load(model):
    import mlx.core as mx

    for layer in model.model.layers:
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

        # free original weights to keep peak memory <25GB
        mlp.gate_proj.weight = None
        mlp.gate_proj.scales = None
        mlp.gate_proj.biases = None
        mlp.up_proj.weight = None
        mlp.up_proj.scales = None
        mlp.up_proj.biases = None
