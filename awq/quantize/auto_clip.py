import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc

__all__ = ["auto_clip_block"]


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    '''
    基于给定的w，通过网格搜索来找到w中ci通道按group_size为单位找到的最佳截断值（clip） 返回的形状是(co,n_group,1)
    '''
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token] #将Input_feat 的形状进一步变成[1, n_sample, n_group, group size] 进行抽样
    w = w.reshape(w.shape[0], 1, -1, group_size) #(co,1,n_group,group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM 用来分割w的co维度 下面进行逐元素乘法的时候并不需要计算完整的w与input的乘积结果。目标是计算每个group的最佳clip
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size] #每次取oc_batch_size个权重  w.shape=(oc_batch,1,n_group,group_size)

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # (oc_batch, 1, n_group, 1) 找到w每个oc_batch通道中的每个group的原始最大值，基准clip

        best_max_val = org_max_val.clone() #(oc_batch, 1, n_group, 1) 用来记录当前oc_batch个权重里的每个group的最佳clip值
        min_errs = torch.ones_like(org_max_val) * 1e9 #(oc_batch, 1, n_group, 1)
        input_feat = input_feat.to(w.device)# (1, n_sample, n_group, group size)

        #【对于sum(dim=-1)的解析】：这一步是对input(n,ci)@w^T(ci,co)的一个模拟计算，但是有很多防止OOM的细节值得进行套讨论。
        #首先是对于两个矩阵相乘的数学表示 假设有矩阵A(mxn) B(nxl) 若C=A@B 则C[i,j]=Σ_{k=0}^{n-1} A[i,k] x B[k,j]
        #对应地设现有W(out,in) Input(n,in) 那么 Output=Input@W Output[i,j]=Σ_{k=0}^{in-1} Input[i,k] x W[k,j]
        #但是我们的目标并不是去计算总的output并以此来计算损失，这么做峰值计算量会很大，可能会OOM。我们的目的是找到每个group中最合适的clip
        #于是对W和Input的形状做了重组。W->(out,1,n_group,grou_size) Input->(1,n,n_group,group_size)  为了方便理解后面再解释此处为什么不是oc_batch而是out，还有此处也没有引入n_sample
        #相乘的结果为Output'(out,n,n_group,group_size)  这个时候如果对最后两个维度分别求和就得到了初始的W 和 Input矩阵相乘的结果
        #也就是 Output[i,j]=Σ_{group=0}^{n_group-1}Σ_{offset=0}^{group_size-1} Input[i,group x 32+offset] x W[group x 32+offset,j]
        #                 = Σ_{k=0}^{in-1} Input[i,k] x W[k,j] 两者是等价的。可以动手写一下方便理解。
        #但是上面也说了我们目标并不是去求原始的Output 所以就只针对最后一个维度进行了sum操作。此时的形状变成了(out,n,n_group) 这是对原始矩阵乘法的一种近似表达，如果此时再对n_group维度进行sum就得到一样的结果了
        #此时结算的结果是以组为单位的，正好契合了我们针对每个group来算最佳clip的需求。而且还免去了在进行sum的计算量
        # 接下来讨论一下防止OOM的优化 ①对于Input我们并不是计算了完整的n 而是对n进行了均匀的采样。所以实际输入的Input是(1, n_sample, n_group, group size)
        #② 我们还将w的形状变成了(oc_batch,1,n_group,group_size) ，也没有一下将全部的out维度输入。基于以上两点我们得到的最终org_out形状就是下面的样子了

        org_out = (input_feat * w).sum(dim=-1)  # (oc_batch_size,n_sample,n_group)=((oc_batch,1,n_group,group_size)*(1, n_sample, n_group, group size)).sum(-1)

        for i_s in range(int(max_shrink * n_grid)): #grid search loop
            max_val = org_max_val * (1 - i_s / n_grid) #随着循环的进行，max_val在逐渐缩小,也就是在逐渐shrink每个group中的最大值 (oc_batch_size, 1, n_group, 1)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val) #将w范围限制在 [-max_val,max_val]之间 (oc_batch,1,n_group,group_size)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config) #对裁剪后的w进行伪量化
            cur_out = (input_feat * q_w).sum(dim=-1) #oc_batch_size,n_sample,n_group
            #【为什么要进行mean(dim=1)呢？】这一步我们对n_sample维度求了平均，是为了平衡不同token可能含有的噪声。如果dim=0会混合不同神经元的误差；dim=2会混合不同分组的误差特性
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape) # (oc_batch, 1, n_group, 1) 沿着n_sample维度求平均 为了使形状匹配min_errs
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs #cur_best_idx.shape (oc_batch, 1, n_group, 1)与err一致，
                                            # 其本质是一个掩码 里边存放的全是True or Fales 如果err对应位置比min_err小就为True
            min_errs[cur_best_idx] = err[cur_best_idx] #将min_err更新，存放损失较小的那一个
            best_max_val[cur_best_idx] = max_val[cur_best_idx] #将本次网格搜索中减少了损失的、缩放后的权重最大值（以group为单位）保存下来 (oc_batch_size, 1, n_group, 1)
        best_max_val_all.append(best_max_val) #每次将 oc_batch 个权重的最佳缩放值存起来

    best_max_val = torch.cat(best_max_val_all, dim=0) #沿着oc_batch维度拼接起来，最终形状(w.shape[0],1,n_group,1)=(co,1,n_group,1)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1) # (co,n_group,1) 返回的结果为将W的ci按group分组后每一组对应地最大值（max_val），即当前组中不会有比其还大的数了，高于此数就截断。


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat):
    '''
    找到module中带名字的线性层权重（除了注意力中的k q两个线性层）的最佳clip值，以group_size为单位查找。
    返回一个list，其中包含若干元组，每个元组格式为(str:name tuple:max_val->(co,n_group,1) )
    '''
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely 通常对于一般mlp有相对稳定的激活分布，分布形状可预测，故clip影响不会太大。但是QK 点积：分布极度依赖输入内容。分布变化巨大！
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        clip_list.append((name, max_val))
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    '''
    对给定的module 将其中的线性层进行clip操作（注意力机制中的k q 两个线性映射层除外）
    '''
    from ..utils.module import get_op_by_name

    for name, max_val in clip_list:#clip_list.shape (str:name tuple:max_val->(co,n_group,1) )
        layer = get_op_by_name(module, name) #拿到具体的线性层
        layer.cuda()
        max_val = max_val.to(layer.weight.device).to(layer.weight.dtype)#(co,n_group,1)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)#改变拿到的线性层的权重形状 (co,ci)-->(co,n_group,group_size)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)#将每一个group都按照对应地best_max_val 来进行clip
        layer.weight.data = layer.weight.data.reshape(org_shape)#还原w形状
        layer.cpu()
