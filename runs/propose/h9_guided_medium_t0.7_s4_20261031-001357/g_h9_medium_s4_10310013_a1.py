"""v1_chunked_sdpa — chunk queries to stay in fused SDPA kernel for short decoding passes."""

NAME = "g_h9_medium_s4_10310013_a1"
EXACT = True
DESCRIPTION = "Chunk 2-8 token decoding queries to max 2 tokens to use fused scaled_dot_product_attention kernel. One-token and prompt passes unchanged."

def apply():
    import mlx.core as mx
    from mlx_lm.models import muse_glimmer as mg
    from mlx_lm.models.base import scaled_dot_product_attention

    orig_call = mg.Attention.__call__

    def new_call(self, x, mask=None, cache=None):
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

        # Only change 2-8 token decoding passes
        if cache is not None and 2 <= L <= 8:
            ratio = self.n_heads // self.n_kv_heads
            max_q_len = 32 // ratio  # 2 for 32/2 heads
            if L > max_q_len:
                outs = []
                start = 0
                while start < L:
                    chunk_len = max_q_len if start + max_q_len < L else L - start
                    q_chunk = q[:, :, start:start + chunk_len, :]
                    if isinstance(mask, mx.array):
                        mask_chunk = mask[..., start:start + chunk_len, :]
                    else:
                        mask_chunk = mask
                    out_chunk = scaled_dot_product_attention(
                        q_chunk, k, v, cache=cache, scale=self.scale, mask=mask_chunk
                    )
                    outs.append(out_chunk)
                    start += chunk_len
                out = mx.concatenate(outs, axis=2)
            else:
                out = scaled_dot_product_attention(
                    q, k, v, cache=cache, scale=self.scale, mask=mask
                )
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = new_call


def post_load(model):
    pass
