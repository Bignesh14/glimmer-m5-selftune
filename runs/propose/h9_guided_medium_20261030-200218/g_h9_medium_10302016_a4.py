"""v1_seq_split — split query sequence for short decoding passes to hit fused SDPA."""

NAME = "g_h9_medium_10302016_a4"
EXACT = True
DESCRIPTION = "Split query sequence into <=2 token chunks for 2-8 token decoding passes to satisfy fused SDPA condition."

def apply():
    import mlx_lm.models.muse_glimmer as mg
    from mlx_lm.models.base import scaled_dot_product_attention
    import mlx.core as mx

    _orig_call = mg.Attention.__call__

    def _patched_call(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        q = mx.fast.rms_norm(q, None, self.eps) * self.qk_scale_factor
        k = mx.fast.rms_norm(k, None, self.eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Only modify short decoding passes; keep 1-token and prompt processing bit-identical
        if cache is not None and 2 <= L <= 8:
            n_heads = self.n_heads
            n_kv = self.n_kv_heads
            # fused kernel condition: L_chunk * (n_heads / n_kv) <= 32
            max_chunk_len = 32 * n_kv // n_heads
            if max_chunk_len < 1:
                max_chunk_len = 1
            if max_chunk_len >= L:
                out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            else:
                # split along sequence dimension to keep each chunk fused
                outs = []
                # slice mask if it is an array
                mask_slices = None
                if isinstance(mask, mx.array):
                    # mask shape (..., L, K)
                    # we will slice the L dimension
                    pass
                for start in range(0, L, max_chunk_len):
                    end = min(start + max_chunk_len, L)
                    q_chunk = q[:, :, start:end, :]
                    # mask handling for array masks
                    if isinstance(mask, mx.array):
                        # mask last two dims are (L, K); slice first of them
                        # assume mask shape (..., L, K)
                        # we need to slice along axis -2
                        # Use dynamic slicing via indexing
                        # For simplicity, create a view by slicing with python indexing on the array
                        # This works because mask is small in L
                        mask_chunk = mask[..., start:end, :]
                    else:
                        mask_chunk = mask
                    out_chunk = scaled_dot_product_attention(q_chunk, k, v, cache=cache, scale=self.scale, mask=mask_chunk)
                    outs.append(out_chunk)
                out = mx.concatenate(outs, axis=2)
        else:
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = _patched_call


def post_load(model):
    pass
