import torch
from block_sparse_attn import block_sparse_attn_func
from einops import rearrange
from block_sparse_attn.bert_padding import pad_input, unpad_input
from .attn_pooling_kernel import attn_with_pooling
from ..utils.gilbert3d import gilbert3d
############parameters###############
# 这里的参数是根据实际情况设置的，可能需要根据具体任务进行调整
use_rearrange=True
max_retain_ratio=0.17
min_retain_ratio=0.05
width=52
height=30
depth=21
sample_gap=30
text_length=0
#####################################
import torch.nn as nn
from torch.nn import functional as F
from ..utils.tools import timeit
def standard_attn(q,k,v):
    mask=torch.ones([q.size(0),q.size(1),q.size(2)//128+1,k.size(2)//128+1],device=q.device,dtype=torch.bool)
    out,lse= block_sparse_attn(q, k, v,block_mask=mask)
    return out,lse
def pad_to_multiple(x, multiple):
    """
    在序列维度（dim=2）上填充 x，使其长度为 multiple 的倍数。
    x: [B, H, L, D]
    """
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        # 对序列维度在后面补 pad_len 个零（注意 F.pad 参数顺序：最后两个数字对应 dim=2 的左右补充）
        x = F.pad(x, (0, 0, 0, pad_len),mode='replicate')
    return x
def random_sample_tokens(x, block_size=64, sample_num=8):
    """
    对输入 x (shape: [B, H, L, D]) 每 block_size 个 token 分为一块，
    在每个块中随机采样 sample_num 个 token。
    要求 L 是 block_size 的倍数。
    返回采样后的结果，形状为 [B, H, num_blocks * sample_num, D]
    """
    B, H, L, D = x.size()
    num_blocks = L // block_size
    # 重塑为 [B, H, num_blocks, block_size, D]
    x_blocks = x.view(B, H, num_blocks, block_size, D)
    
    # 为每个块生成随机数，并用 topk 选出 sample_num 个随机索引
    rand_vals = torch.rand(B, H, 1, block_size, device=x.device)
    _, indices = torch.topk(rand_vals, sample_num, dim=3)
    # indices 的 shape: [B, H, num_blocks, sample_num]
    
    # 将 indices 扩展到与 x_blocks 最后一个维度 D 对齐
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_blocks, -1, D)
    # 利用 torch.gather 从每个块中采样 token
    sampled = torch.gather(x_blocks, 3, indices_expanded)
    # 重塑回 [B, H, num_blocks * sample_num, D]
    sampled = sampled.view(B, H, num_blocks * sample_num, D)
    return sampled
# @timeit
def efficient_attn_with_pooling(q, k, v, block_size=128, num_keep=32):
    """
    计算下采样后的注意力池化：
      - 从 q, k (shape: [B, H, seq, D]) 中，从 token 224 开始，每 64 个 token 随机采样 num_keep 个 token。
      - 对采样后的 q, k 计算注意力，并用卷积实现下采样，使得效果等效于对原始 attn 做了 64×64 的求和池化。
      - 对 q 与 k 在序列维度做 padding，保证采样时不会丢失末尾数据。
      
    参数：
      block_size: 每个块的 token 数（固定为 64）
      num_keep: 每个块中保留的 token 数（默认 8，可以自定义）
    """
    # 取从 token 224 开始的部分
    q_ = q[:, :, :, :]
    k_ = k[:, :, :, :]
    
    # 在序列维度上 padding 至 block_size 的倍数
    q_ = pad_to_multiple(q_, block_size)
    k_ = pad_to_multiple(k_, block_size)
    
    # 对 q 和 k 分块并随机采样
    sampled_q = random_sample_tokens(q_, block_size, num_keep)  # [B, H, num_blocks*num_keep, D]
    sampled_k = random_sample_tokens(k_, block_size, num_keep)
    _,pooling= attn_with_pooling(
        sampled_q, sampled_k, v, False, 1.0 / (sampled_q.size(-1) ** 0.5), num_keep
    )
    return pooling
def simple_pooling(x, sample_gap=sample_gap):
    x=pad_to_multiple(x, sample_gap)
    x=x.reshape(x.size(0), x.size(1), -1, sample_gap, x.size(-1))
    x=x.mean(dim=-2)
    x=x.reshape(x.size(0), x.size(1), -1, x.size(-1))
    return x
# @timeit
def org_attn_with_pooling(q, k, v):
    sm_scale=1.0 / (q.size(-1) ** 0.5)
    causal=False
    block_size=128
    _, pooling = attn_with_pooling(q, k, v, causal, sm_scale,block_size)
    return pooling

class GilbertRearranger(nn.Module):
    """基于 Gilbert 曲线的序列重排器，用于视频和文本数据的重新排列。"""
    def __init__(self, width, height, depth, text_length=224):
        super(GilbertRearranger, self).__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.total_elements = width * height * depth
        self.text_length = text_length

        coord_to_index = self._gilbert3d_with_index(width, height, depth)
        original_order2gilbert_order = [0] * self.total_elements
        gilbert_order2original_order = [0] * self.total_elements

        for coord_idx, org_idx in coord_to_index.items():
            original_order2gilbert_order[org_idx] = coord_idx
            gilbert_order2original_order[coord_idx] = org_idx

        # self.original_order2gilbert_order = torch.tensor(original_order2gilbert_order, dtype=torch.long)
        # self.gilbert_order2original_order = torch.tensor(gilbert_order2original_order, dtype=torch.long)
        self.register_buffer(
            "original_order2gilbert_order",
            torch.tensor(original_order2gilbert_order, dtype=torch.long),
        )
        self.register_buffer(
            "gilbert_order2original_order",
            torch.tensor(gilbert_order2original_order, dtype=torch.long),
        )

    def _gilbert3d_with_index(self, width, height, depth):
        """生成 Gilbert 曲线的坐标到索引映射。"""
        coord_to_index = {}
        index = 0
        def coord_to_index_func(x, y, z):
            return x + width * (y + height * z)
        for x, y, z in gilbert3d(width, height, depth):
            coord_index = coord_to_index_func(x, y, z)
            coord_to_index[coord_index] = index
            index += 1
        return coord_to_index
    def rearrange(self, q, k, v):
        """将 q、k、v 张量的视频部分按 Gilbert 曲线顺序重排。"""
        seq_dim = -2

        q_rearranged = q.index_select(seq_dim, self.original_order2gilbert_order)
        k_rearranged = k.index_select(seq_dim, self.original_order2gilbert_order)
        v_rearranged = v.index_select(seq_dim, self.original_order2gilbert_order)

        return (q_rearranged,
                k_rearranged,
                v_rearranged)
    
    def reversed_rearrange(self, out):
        """将输出张量的视频部分从 Gilbert 曲线顺序恢复到原始顺序。"""
        seq_dim = -2
        video_part=out
        out_reversed = video_part.index_select(seq_dim, self.gilbert_order2original_order)
        return out_reversed
    
# @timeit
def transfer_attn_to_mask(attn, mode="energy", init_k=None, max_retain_ratio=0.7, min_retain_ratio=0.1, energy_threshold=0.95):
    """
    将注意力权重转换为掩码矩阵。

    Args:
        attn (torch.Tensor): 注意力权重矩阵，形状为 [batch, head, seq, seq]
        mode (str): 掩码生成模式，支持 "topk" 或 "energy"
        init_k (float, optional): topk 模式下的初始 k 值
        max_retain_ratio (float): energy 模式下的最大保留比例
        min_retain_ratio (float): energy 模式下的最小保留比例
        energy_threshold (float): energy 模式下的能量阈值

    Returns:
        torch.Tensor: 二值掩码矩阵，形状同输入
    """
    def get_fix_weight():
        max_exp=5
        weight_map_temp= (torch.arange(attn.shape[-1], dtype=torch.int64).to(attn.device).reshape(1,1,1,-1)-torch.arange(attn.shape[-2], dtype=torch.int64).to(attn.device).reshape(1,1,-1,1))/attn.shape[-1]*max_exp
        weight_map=torch.clamp(torch.exp2(weight_map_temp.abs()),min=0,max=2)
        diag_indices = torch.arange(weight_map.size(-2), device=weight_map.device)
        weight_map[:, :, diag_indices, diag_indices] = 2
        return weight_map
    batch, heads, seq, _ = attn.shape
    device = attn.device
    mask = torch.zeros_like(attn, dtype=torch.bool)
    if not hasattr(transfer_attn_to_mask, "fix_weight"):
            transfer_attn_to_mask.fix_weight = get_fix_weight()
    # print("fix_weight.shape",transfer_attn_to_mask.fix_weight.shape,"attn.shape",attn.shape)
    # attn=attn*transfer_attn_to_mask.fix_weight

    if mode == "topk":
        if init_k is None:
            raise ValueError("在 topk 模式下必须提供 init_k")
        init_k = int(seq * init_k) if init_k < 1 else int(init_k)
        sorted_attn, indices = torch.sort(attn, dim=-1, descending=True)
        cum_energy = torch.cumsum(sorted_attn, dim=-1)
        total_energy = cum_energy[..., -1:]
        current_k = torch.full((batch, heads, seq), init_k, device=device, dtype=torch.int64)

        current_energy = cum_energy.gather(dim=-1, index=(current_k - 1).unsqueeze(-1)).squeeze(-1)
        condition_met = current_energy >= (0.6 * total_energy.squeeze(-1))
        condition_met1 = current_energy >= (0.9 * total_energy.squeeze(-1))

        need_update = (~condition_met) & (current_k < seq)
        need_update1 = (~condition_met1) & (current_k < seq)
        current_k[need_update] = torch.clamp(current_k[need_update] * 3, max=seq)
        current_k[need_update1] = torch.clamp(current_k[need_update1] // 3 * 2, max=seq)

        pos_indices = torch.arange(seq, device=device).view(1, 1, 1, seq)
        keep_mask = pos_indices < current_k.unsqueeze(-1)
        mask.scatter_(-1, indices, keep_mask)

    elif mode == "energy":
        min_retain = max(1, int(seq * min_retain_ratio))
        max_retain = max(1, int(seq * max_retain_ratio))
        sorted_attn, indices = torch.sort(attn, dim=-1, descending=True)
        cum_energy = torch.cumsum(sorted_attn, dim=-1)
        total_energy = cum_energy[..., -1:]

        energy_mask = cum_energy >= energy_threshold * total_energy
        k_indices = torch.argmax(energy_mask.int(), dim=-1)
        unsatisfied = (cum_energy[..., -1:] < energy_threshold * total_energy).squeeze(-1)
        k_indices = torch.where(unsatisfied, seq, k_indices)
        k_indices = torch.clamp(k_indices, min=min_retain, max=max_retain)

        pos_indices = torch.arange(seq, device=device).view(1, 1, 1, seq)
        keep_mask = pos_indices < k_indices.unsqueeze(-1)
        mask.scatter_(-1, indices, keep_mask)

    else:
        raise ValueError(f"不支持的模式: {mode}")
    return mask


def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None):
    """
    Convert q, k, v tensors for block-sparse attention.
    Args:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen_q), bool, optional
        key_padding_mask: (batch_size, seqlen_k), bool, optional
    Returns:
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v, output_pad_fn, dq_pad_fn, dk_pad_fn
    """
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q)
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k.device)
        max_seqlen_k = seqlen_k

    dq_pad_fn = output_pad_fn
    dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size) if key_padding_mask is None else pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
    
    return (
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
        q, k, v, output_pad_fn, dq_pad_fn, dk_pad_fn
    )
def block_sparse_attn(q, k, v, block_mask):
    #print("q.shape:", q.shape, "\nk.shape:", k.shape, "\nv.shape:", v.shape, "\nblock_mask.shape:", block_mask.shape)
    """
    Block-sparse attention mechanism.
    Args:
        q: (batch_size, nheads, seqlen, d)
        k: (batch_size, nheads, seqlen, d)
        v: (batch_size, nheads, seqlen, d)
        block_mask: (batch_size, nheads, blocks_q, blocks_k), bool
    Returns:
        out: (batch_size, nheads, seqlen, d)
    """
    batch_size, nheads, seqlen, d = q.shape
    device = q.device
    query_padding_mask = torch.ones(batch_size, q.shape[2], dtype=torch.bool, device=device)
    key_padding_mask = torch.ones(batch_size, k.shape[2], dtype=torch.bool, device=device)
    q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, _, _, _, output_pad_fn, _, _ = generate_qkv(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), query_padding_mask, key_padding_mask
    )
    base_blockmask = block_mask.contiguous()
    head_mask_type = torch.ones(nheads, dtype=torch.int32, device=device)
    streaming_info = torch.zeros(2 * nheads, dtype=torch.int32, device=device)
    #print(f"q_unpad shape: {q_unpad.shape}, k_unpad shape: {k_unpad.shape}, v_unpad shape: {v_unpad.shape}")
    out_unpad,lse,_ = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask,
        max_seqlen_q, max_seqlen_k, p_dropout=0.0, deterministic=True, softmax_scale=None, is_causal=False,
        exact_streaming=False, return_attn_probs=True
    )
    out = output_pad_fn(out_unpad)
    out = out.view(batch_size, q.shape[2], nheads, d)
    out = out.permute(0, 2, 1, 3)
    return out,lse.unsqueeze(-1).to(q.device).to(q.dtype)  # Remove the last dimension for lse

def adaptive_block_sparse_attn(q, k, v):
    """
    Adaptive block-sparse attention mechanism.
    Creates a block mask automatically (based on q, k) without gradient tracking for mask steps.
    Args:
        q: (batch_size, nheads, seqlen, d)
        k: (batch_size, nheads, seqlen, d)
        v: (batch_size, nheads, seqlen, d)
    Returns:
        out: (batch_size, nheads, seqlen, d)
    """
    global max_retain_ratio, min_retain_ratio
    causal = False
    sm_scale = 1.0 / (q.size(-1) ** 0.5)
    block_size = 128
    # Disable gradient tracking for pooling and mask operations
    with torch.no_grad():
        # sparsity_ref=judge_sparsity(q,k)
        # sparsity_ref=sparsity_ref/sparsity_ref.mean()
        # max_retain_ratio_ = torch.clamp(max_retain_ratio * sparsity_ref, min=0.05, max=0.9)
        # max_retain_ratio_ = torch.ones([q.size(0),q.size(1)],device=q.device)*max_retain_ratio
        # min_retain_ratio_ = torch.ones([q.size(0),q.size(1)],device=q.device)*min_retain_ratio
        pooling = efficient_attn_with_pooling(q, k, v,block_size=block_size)
        
        mask = transfer_attn_to_mask(
            pooling,
            mode="energy",
            init_k=None,
            max_retain_ratio=max_retain_ratio,
            min_retain_ratio=min_retain_ratio,
            energy_threshold=0.95
        )
    out1,lse1 = block_sparse_attn(q, k, v, mask)
    k_pooling=simple_pooling(k, sample_gap=sample_gap)
    v_pooling=simple_pooling(v, sample_gap=sample_gap)
    # print("*"*20)
    # print(f"q shape: {q.shape}, k_pooling shape: {k_pooling.shape}, v_pooling shape: {v_pooling.shape}\n,q_dtype: {q.dtype}, k_pooling dtype: {k_pooling.dtype}, v_pooling dtype: {v_pooling.dtype}")
    out2,lse2 = standard_attn(q, k_pooling, v_pooling)
    # print("out1_dtype:", out1.dtype, "out2_dtype:", out2.dtype)
    # 数值稳定的 alpha 计算
    sample_gap_tensor = torch.tensor(sample_gap, device=lse1.device, dtype=lse1.dtype)
    log_sample_gap = torch.log(sample_gap_tensor)
    
    # 计算 log 空间的权重
    log_weight1 = lse1
    log_weight2 = lse2 + log_sample_gap
    
    # 使用 logsumexp 技巧进行数值稳定的 softmax
    max_log_weight = torch.maximum(log_weight1, log_weight2)
    
    exp1 = torch.exp(log_weight1 - max_log_weight)
    exp2 = torch.exp(log_weight2 - max_log_weight)
    
    alpha = exp1 / (exp1 + exp2)
    
    # 方法2：使用 PyTorch 内置的数值稳定函数（推荐）
    # log_weights = torch.stack([lse1, lse2 + log_sample_gap], dim=-1)
    # alpha = torch.softmax(log_weights, dim=-1)[..., 0:1]
    
    out = out1 * alpha + out2 * (1 - alpha)
    #print("out.mean:", out.abs().mean().item(), "out1.mean:", out1.abs().mean().item(), "out2.mean:", out2.abs().mean().item())
    return out,1-mask.float().mean()-1/sample_gap  # Return the average sparsity as a float


class AdaptiveBlockSparseAttnTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gilbert_rearranger = GilbertRearranger(width, height, depth, text_length)
        self.sparsity_acc = 0.0  # Accumulator for sparsity sum
        self.sparsity_counter = 0  # Counter for number of updates
        self.use_rearrange=use_rearrange
    # @timeit
    def forward(self, q, k, v):
        if(self.use_rearrange):
            q_r, k_r, v_r = self.gilbert_rearranger.rearrange(q, k, v)
        else:
            q_r=q
            k_r=k
            v_r=v
        # Compute block-sparse attention and get sparsity
        timestep=(self.sparsity_counter%(30*8))//30
        if(timestep<=-1):
            out_r=torch.nn.functional.scaled_dot_product_attention(q_r, k_r, v_r)
            sparsity=torch.tensor(0)
        else:
            out_r, sparsity = adaptive_block_sparse_attn(q_r, k_r, v_r)
        # Update sparsity statistics
        self.sparsity_acc += sparsity.item()  # Convert tensor to float
        self.sparsity_counter += 1
        # Print average sparsity every 8 calls
        if self.sparsity_counter % 200 == 0:
            avg_sparsity = self.sparsity_acc / self.sparsity_counter
            print(f"sparsity: {avg_sparsity}")
        # Reverse the arrangement
        if(self.use_rearrange):
            out = self.gilbert_rearranger.reversed_rearrange(out_r)
        else:
            out=out_r
        return out