"""v_chunked_sdpa — split 3-8 token queries into ≤2 token chunks to hit the fused SDPA kernel."""

NAME = "g_h9_medium_s1_10302354_a1"
EXACT = True
DESCRIPTION = "Split 2-8 token generation queries into chunks of size ≤2 so query_length * (n_heads/n_kv_heads) ≤32 and the fused vector SDPA kernel is used. One-token and prompt processing are unchanged."

def apply():
    import mlx.core as mx
    from mlx_lm.models import muse_glimmer as mg
    from mlx_lm.models.base import scaled_dot_product_attention

    orig_Attention = mg.Attention

    class PatchedAttention(orig_Attention):
        def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
            B, L, _ = x.shape

            # ---- original pre-SDPA work ----
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

            # ---- chunking for the fused kernel ----
            # fused kernel requires query_length * (n_heads/n_kv_heads) <= 32
            # n_heads/n_kv_heads = 16  -> max query_length = 2
            use_chunk = cache is not None and 2 < L <= 8
            if use_chunk:
                if cache is not None:
                    k, v = cache.update_and_fetch(k, v)

                chunk_size = 2  # max that satisfies 2*16 <=32
                out_chunks = []
                # keep keys/values for the whole current batch so later chunks can attend to earlier ones
                for start in range(0, L, chunk_size):
                    end = start + chunk_size
                    if end > L:
                        end = L
                    q_chunk = q[:, :, start:end, :]

                    # mask slicing for sliding-window arrays
                    if isinstance(mask, mx.array):
                        # mask shape (..., L, K)
                        mask_chunk = mask[..., start:end, :]
                        out_chunk = scaled_dot_product_attention(
                            q_chunk, k, v, cache=cache, scale=self.scale, mask=mask_chunk
                        )
                    else:
                        out_chunk = scaled_dot_product_attention(
                            q_chunk, k, v, cache=cache, scale=self.scale, mask=mask
                        )
                    out_chunks.append(out_chunk)
                out = mx.concatenate(out_chunks, axis=2)
            else:
                if cache is not None:
                    k, v = cache.update_and_fetch(k, v)
                out = scaled_dot_product_attention(
                    q, k, v, cache=cache, scale=self.scale, mask=mask
                )

            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            out = out * mx.sigmoid(self.gate_proj(x))
            return self.o_proj(out)

    mg.Attention = PatchedAttention


def post_load(model):
    pass
