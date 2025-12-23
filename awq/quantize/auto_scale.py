import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer

from .qmodule import ScaledActivation
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x): #函数将输入展平后计算每个特征维度的平均绝对激活值，输出形状从 (n, in_feature)变为 (in_feature,)。
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales): #保持无损输出
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)
    # 因为layerNorm中默认开启了elementwise_affine，所以也是有对应的weight 和 bias的，一般都有权重，bias根据不同的layerNorm变体而可能不会有
    ln.weight.div_(scales)  #将layerNorm的权重÷ scales 后面再通过让全连接层权× scales 实现无损输出（因为是线性缩放）
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device).to(fc1.weight.dtype)

    # fc1.weight.div_(scales.view(-1, 1)) 解释一下：scales.size(0)为fc2.in_feature 这刚好是fc.out_feature 所以fc1.weight[-scales.size(0) :]是将所有行÷ scales 负号是从后往前数，不过一般来说两个维度是一样的，这么做可能是为了处理一些特殊情况。
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))# 此处的表述比较奇怪，但是其本质就是让fc1的权重÷scales 再让fc2的权重×scales
                                                    # 不论是注意力机制的 v 和 out两个ffn 还是 瓶颈结构的先升后降ffn ,fc1.out_feature == fc2.in_feature 虽然蹩脚，但没问题
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)
    #因为此时的scales已经是 ScaledActivation，其在进行forward的时候已经÷了scales了所以不需要再÷一遍了，直接让线性层×scales就行了。
    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat):
    '''
    返回一个list,其中包含了若干元组。每个元组格式为(str: name,tuple: (name_of_layer1,name_of_layer2...name_of_layern),scales ->(in_feature))
    如果是注意力层元组中第二个元素就是k q v 三个线性映射曾的名字 否则就是单个线性层的名字
    '''
    from .quantizer import pseudo_quantize_tensor

    # firstly, get the weight quantize function
    if w_bit is not None:

        def w_quantize_func(p): #使用伪量化拿到量化后又反量化的权重
            return pseudo_quantize_tensor(
                p,
                n_bit=w_bit,
                **q_config,
            ).detach()

    else:

        def w_quantize_func(p): #若没有提供量化位宽，直接返回原始的权重。可能是为了进行消融实验
            return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        '''
        搜索给定block中linears2scale的最佳缩放因子，block可以是一个完整的块也可以是一个完整块中的一部分 e.g. block=module.self_attn
        '''
        # w: co, ci
        # x: n, ci  此处是为了方便理解，实际上输入的x形状是(n_split,block_size,in_feature)
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs) #计算原始未被缩放权重下的块输出
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x) #拿到激活值各输入通道均值 （in_feature,）

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()} #state_dict保存了当前block中全部的参数值（value） 使用self.xxx 作为key
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid #ratio从0, 0.05, 0.10, 0.15, ..., 0.95 逐渐增加 此处对应论文中alpha:  ---> s^α
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)# 缩放因子来自于激活值，换句话说缩放因子对于激活值敏感！
            scales = scales / (scales.max() * scales.min()).sqrt() #使缩放因子在对数空间对称分布，保持数值范围的平衡
            for fc in linears2scale:
                #要注意！此处的scales是由激活值x产生，而与权重w无关，这里的缩放因子对应的是论文中的's'
                #而不是零点量化中的 scales 那个缩放因子是来自于权重w，与激活值是无关的。不要混淆了。
                #此处先乘后除的操作是： 找 s，使得 Q(W / s) * s ≈ W 。但对激活 A 敏感：min || (W * A) - (s * Q(W / s) * A) ||^2≈ min MSE(out, org_out)。
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device)) #临时 W' = W * s，"放大" 权重通道，使后续伪量化"看到" 调整后分布。（inplace操作避免 copy 大 tensor）
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1)) #注入量化噪声后"还原"权重
            out = block(x, **kwargs) #拿到权重缩放之后的块输出
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best: #记录最佳的ratio和缩放因子
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd) #复原block的权重 进入新一轮的搜索
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1) #(in_feature,)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()#将最接近原初输出结果 的scale返回

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        '''
        返回一个元组。包含了使用哪个prev_op的激活 → 哪些 layers 用此 scale → scale 值。
        '''
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None: # 如果module2inspect=None 则使用一个具体的层而不是整个block来计算函数_search_module_scale中的loss 单层级别的计算，结果是局部误差，
                                    # 传入如果module2inspect 则是模块级别的计算 端到端的误差
            assert len(layers) == 1 # 若不指定 module2inspect 那么只能处理单层的情况，计算的结果也是单层的scale
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op), #返回 prev_op_name 是为了标识 scales 对应的激活来源和应用位置；
            tuple([get_op_name(module, m) for m in layers]),
            scales, #(in_feature,)
        )

    scales_list = []  # return the searched scales

    if isinstance(module, OPTDecoderLayer):
        # attention input 对于QKV三个线性映射层进行缩放
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out 对注意力合并线性映射层进行缩放 此时没有指定module2inspect 意味着使用module.self_attn.out_proj本身计算org_out 来计算loss
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif isinstance(module, (LlamaDecoderLayer, Qwen2DecoderLayer)):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )

        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "falcon" in str(module.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.mlp.dense_h_to_4h,
                        module.self_attention.query_key_value,
                    ],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn,
                    layers=[module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp,
                    layers=[module.mlp.dense_h_to_4h],
                    inp=input_feat["mlp.dense_h_to_4h"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )
    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    '''
    对scales_list中的每个 layer应用无损缩放：前一层除以scales 后一层乘以scales 既保护了salient weight 又数学等价，提前应用scales减少计算
    '''
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda() #scales 的形状(in_feature,) 具体根据线性层不同而不同

        if isinstance(prev_op, nn.Linear): #如果前一层是线性层 e.g. prev_op=module.self_attn.v_proj
            assert len(layers) == 1 #那么当前要缩放的层也得是单个线性层  e.g. layers=[module.self_attn.out_proj]
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm)): #同样地，对layer Norm进行输出无损缩放
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation, nn.SiLU)):
            new_module = ScaledActivation(prev_op, scales)#给GELU加上能训练的参数
            set_op_by_name(module, prev_op_name, new_module) #替换激活函数为缩放版本的GELU
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        # 将input也同步缩小，这与上面的权重缩放不冲突，上面是针对inference 这里是为了进行clip操作针对calib data进行缩放
        #因为上面已经对w乘上了scale了，下面对input÷scales是为了在进行clip操作的时候保持计算值的一致方便计算err。在clip操作的时候需要input与w相乘。此处是为了保持计算的一致。
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
