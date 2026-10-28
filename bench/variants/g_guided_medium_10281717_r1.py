"""fuse_mlp_decode_only_inplace — fuse MLP gate/up in-place, decode-only path."""

NAME = "g_guided_medium_10281717_r1"
EXACT = True
DESCRIPTION = "Fuse MLP gate/up in-place, use fused matmul for L==1 only, slice fused weight for prompts."

def apply():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.muse_glimmer import MLP

    _orig_mlp_call = MLP.__call__

    def fused_mlp_call(self, x):
        if not hasattr(self, "gate_up_fused_weight"):
            return _orig_mlp_call(self, x)

        # Decode path: single fused quantized matmul
        if x.shape[1] == 1:
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

        # Prompt path: slice fused weight back to two projections
        # This avoids the 1K slowdown while keeping only one copy of weights
        inter = self.mlp_intermediate_size
        # slice fused buffers
        w_gate = self.gate_up_fused_weight[:inter]
        w_up = self.gate_up_fused_weight[inter:]
        s_gate = self.gate_up_fused_scales[:inter]
        s_up = self.gate_up_fused_scales[inter:]
        b_gate = self.gate_up_fused_biases[:inter]
        b_up = self.gate_up_fused_biases[inter:]

        gate = mx.quantized_matmul(
            x,
            w_gate,
            scales=s_gate,
            biases=b_gate,
            transpose=True,
            group_size=self.gate_up_group_size,
            bits=self.gate_up_bits,
            mode=self.gate_up_mode,
        )
        up = mx.quantized_matmul(
            x,
            w_up,
            scales=s_up,
            biases=b_up,
            transpose=True,
            group_size=self.gate_up_group_size,
            bits=self.gate_up_bits,
            mode=self.gate_up_mode,
        )
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

        # Build one fused copy, then discard the two originals
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

        # Free originals to keep memory close to baseline
        mlp.gate_proj.weight = None
        mlp.gate_proj.scales = None
        mlp.gate_proj.biases = None
        mlp.up_proj.weight = None
        mlp.up_proj.scales = None
        mlp.up_proj.biases = None
