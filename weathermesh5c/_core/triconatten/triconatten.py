import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=4),
    ],
    key=["M"],
)
@triton.jit
def _attention_kernel(
    Q,
    K,
    V,
    WHICH,
    MASK2,
    O,
    T,
    H,
    M,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_w_b,
    stride_w_m,
    stride_m_t,
    stride_m_m,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h2 = tl.program_id(axis=1)

    b_offs = tl.arange(0, BLOCK_B)
    d_offs = tl.arange(0, BLOCK_D)
    m_offs = tl.arange(0, BLOCK_M)

    base_t = pid_b * BLOCK_B
    t_idx = base_t + b_offs
    b_mask = t_idx < T

    h0 = pid_h2 * 2
    h1 = h0 + 1
    has_h1 = h1 < H

    # Accumulators for head 0
    out_acc0 = tl.zeros([BLOCK_B, BLOCK_D], tl.float32)
    m_i0 = tl.full([BLOCK_B], -float("inf"), tl.float32)
    s_i0 = tl.zeros([BLOCK_B], tl.float32)
    # Accumulators for head 1
    out_acc1 = tl.zeros([BLOCK_B, BLOCK_D], tl.float32)
    m_i1 = tl.full([BLOCK_B], -float("inf"), tl.float32)
    s_i1 = tl.zeros([BLOCK_B], tl.float32)

    # Load Q tiles
    q0_ptrs = Q + t_idx[:, None] * stride_q_t + h0 * stride_q_h + d_offs[None, :] * stride_q_d
    q0 = tl.load(q0_ptrs, mask=b_mask[:, None], other=0.0)
    q1_ptrs = Q + t_idx[:, None] * stride_q_t + h1 * stride_q_h + d_offs[None, :] * stride_q_d
    q1 = tl.load(q1_ptrs, mask=b_mask[:, None] & has_h1, other=0.0)

    for m_start in range(0, M, BLOCK_M):
        mm = m_start + m_offs
        m_mask = mm < M
        # WHICH panel and MASK2 for both heads (shared across heads)
        w_ptrs = WHICH + pid_b * stride_w_b + mm * stride_w_m
        # Load as u32 then cast to i32 so sentinel 0xFFFFFFFF becomes -1
        nbr_idx = tl.load(w_ptrs, mask=m_mask, other=0).to(tl.int32)
        valid = (nbr_idx != -1) & m_mask
        mask_ptrs = MASK2 + t_idx[:, None] * stride_m_t + mm[None, :] * stride_m_m
        row_sel = tl.load(mask_ptrs, mask=b_mask[:, None] & m_mask[None, :], other=False)

        # Head 0
        k0_T = tl.load(
            K + d_offs[:, None] * stride_k_d + h0 * stride_k_h + nbr_idx[None, :] * stride_k_t,
            mask=valid[None, :],
            other=0.0,
        )
        logits0 = tl.dot(q0.to(tl.float16), k0_T.to(tl.float16), out_dtype=tl.float32)
        # Apply both validity and triangular mask before softmax
        masked_logits0 = tl.where((valid[None, :]) & row_sel, logits0, float("-inf"))
        # Per-row predicate: does this panel contain any valid entries?
        panel_has_valid0 = tl.max(((valid[None, :]) & row_sel).to(tl.int32), axis=1) > 0
        block_max0 = tl.max(masked_logits0, axis=1)
        # For rows with fully-masked panels, do not change the running max
        block_max0 = tl.where(panel_has_valid0, block_max0, m_i0)
        m_new0 = tl.maximum(m_i0, block_max0)
        # Compute deltas first, then gate to avoid (-inf) - (-inf)
        delta_m0 = m_i0 - m_new0
        delta_m0 = tl.where(panel_has_valid0, delta_m0, 0.0)
        alpha0 = tl.exp(delta_m0)
        shifted_logits0 = masked_logits0 - m_new0[:, None]
        shifted_logits0 = tl.where(panel_has_valid0[:, None], shifted_logits0, float("-inf"))
        beta0 = tl.exp(shifted_logits0)
        s_i0 = s_i0 * alpha0 + tl.sum(beta0, axis=1)
        v0 = tl.load(
            V + nbr_idx[:, None] * stride_v_t + h0 * stride_v_h + d_offs[None, :] * stride_v_d,
            mask=valid[:, None],
            other=0.0,
        )
        out_acc0 = out_acc0 * alpha0[:, None] + tl.dot(
            beta0.to(tl.float16), v0.to(tl.float16), out_dtype=tl.float32
        )
        m_i0 = m_new0

        # Head 1
        k1_T = tl.load(
            K + d_offs[:, None] * stride_k_d + h1 * stride_k_h + nbr_idx[None, :] * stride_k_t,
            mask=valid[None, :] & has_h1,
            other=0.0,
        )
        logits1 = tl.dot(q1.to(tl.float16), k1_T.to(tl.float16), out_dtype=tl.float32)
        # Apply both validity and triangular mask before softmax
        masked_logits1 = tl.where((valid[None, :]) & row_sel, logits1, float("-inf"))
        # Per-row predicate: does this panel contain any valid entries?
        panel_has_valid1 = tl.max(((valid[None, :]) & row_sel).to(tl.int32), axis=1) > 0
        block_max1 = tl.max(masked_logits1, axis=1)
        # For rows with fully-masked panels, do not change the running max
        block_max1 = tl.where(panel_has_valid1, block_max1, m_i1)
        m_new1 = tl.maximum(m_i1, block_max1)
        # Compute deltas first, then gate to avoid (-inf) - (-inf)
        delta_m1 = m_i1 - m_new1
        delta_m1 = tl.where(panel_has_valid1, delta_m1, 0.0)
        alpha1 = tl.exp(delta_m1)
        shifted_logits1 = masked_logits1 - m_new1[:, None]
        shifted_logits1 = tl.where(panel_has_valid1[:, None], shifted_logits1, float("-inf"))
        beta1 = tl.exp(shifted_logits1)
        s_i1 = s_i1 * alpha1 + tl.sum(beta1, axis=1)
        v1 = tl.load(
            V + nbr_idx[:, None] * stride_v_t + h1 * stride_v_h + d_offs[None, :] * stride_v_d,
            mask=valid[:, None] & has_h1,
            other=0.0,
        )
        out_acc1 = out_acc1 * alpha1[:, None] + tl.dot(
            beta1.to(tl.float16), v1.to(tl.float16), out_dtype=tl.float32
        )
        m_i1 = m_new1

    # Normalize and store outputs
    o0 = (out_acc0 / s_i0[:, None]).to(tl.float16)
    o1 = (out_acc1 / s_i1[:, None]).to(tl.float16)
    o0_ptrs = O + t_idx[:, None] * stride_o_t + h0 * stride_o_h + d_offs[None, :] * stride_o_d
    o1_ptrs = O + t_idx[:, None] * stride_o_t + h1 * stride_o_h + d_offs[None, :] * stride_o_d
    tl.store(o0_ptrs, o0, mask=b_mask[:, None])
    tl.store(o1_ptrs, o1, mask=b_mask[:, None] & has_h1)


def tri_conatten(q, k, v, which, mask2, B=16):
    Nvert, Dv, Hh, Hd = q.shape
    Ttot = Nvert * Dv
    qv = q.view(Ttot, Hh, Hd)
    kv = k.view(Ttot, Hh, Hd)
    vv = v.view(Ttot, Hh, Hd)
    out = torch.empty_like(qv)

    NB = (Ttot + B - 1) // B
    M = which.shape[1]
    grid = (NB, (Hh + 1) // 2)
    _attention_kernel[grid](
        qv,
        kv,
        vv,
        which,
        mask2,
        out,
        Ttot,
        Hh,
        M,
        qv.stride(0),
        qv.stride(1),
        qv.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        vv.stride(0),
        vv.stride(1),
        vv.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        which.stride(0),
        which.stride(1),
        mask2.stride(0),
        mask2.stride(1),
        BLOCK_B=B,
        BLOCK_D=Hd,
    )
    return out.view(Nvert, Dv, Hh * Hd)
