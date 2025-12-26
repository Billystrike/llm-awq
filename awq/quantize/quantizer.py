import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    '''
    将ffn之间的激活函数变成带缩放版本的激活函数,换句话说就是在激活函数的输出上乘以一个可学习的缩放因子
    目前支持的模型有：Bloom、MPT、Falcon、BigCode、NeoX
    '''
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):#如果这个激活函数已经进行过缩放了，那就不用再来一遍了。
            return
        c = module.mlp.dense_h_to_4h.out_features#线性层输出维度
        act = ScaledActivation(#实例化缩放激活函数类
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)#按照name来匹配module中对应地层，替换为带缩放版本的激活函数
    elif "mptblock" in str(module.__class__.__name__).lower():#后面的判断分支其逻辑与上文相同，只不过变成了判断module中是否包含某个字符串这种形式了
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    '''
    假装自己是量化后的模型，但其实还在浮点域里计算。先量化在反量化，本质是为了模拟量化后的噪音
    如果get_scale_zp 返回的scales 和 zero 形状为(out_feature,  in_feature / q_group_size)
    '''
    org_w_shape = w.shape#假设原本的形状是 out_feature * in_feature
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0#确保权重in_feature能够整除量化group size
        w = w.reshape(-1, q_group_size)#修改权重形状，使其符合量化group大小 w.shape=(out_feature * in_feature / q_group_size, q_group_size)
    assert w.dim() == 2
    if zero_point:#使用零点量化
        max_val = w.amax(dim=1, keepdim=True)#返回每一组的权重最大值，形状：(out_feature * in_feature / q_group_size,1)
        min_val = w.amin(dim=1, keepdim=True)#返回每一组的权重最小值
        max_int = 2**n_bit - 1 #nbit 表示的最大值
        min_int = 0 #无符号数最小值为0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int #计算缩放步长Δ 形状：(out_feature * in_feature / q_group_size,1) 
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int) #形状：(out_feature * in_feature / q_group_size,1)
    # we actually never used this
    else:  # we actually never used this 最大绝对值量化适用于对称权重值
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0 #确保缩放因子中没有Nan
    assert torch.isnan(w).sum() == 0#权重同理

    if inplace:#原地修改w节省显存 先进行了量化操作之后有紧接着进行了反量化，所以函数名叫做伪量化。不改变 dtype，只在浮点数上模拟整数量化的误差
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:#逻辑是一样的，只不过没有直接原地修改
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w #返回的w随时浮点数，但数值上已经被离散化到了量化精度对应的水平。常见于量化感知训练（QAT, Quantization-Aware Training）


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model) #拿到当前模型的所有block
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i]) #拿到单个block的所有带名字的线性层
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor( #对线性层权重进行伪量化，模拟其在真实量化情况下的精度损失
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    '''
    先将model中每个线性层激活函数变成缩放版本的激活函数
    若只是init 就将线性层替换成WQLinear，其权重qweight初始为0  scales和scaled_zeros也为0 (n_group,out_feature)
    不只是初始化就先伪量化，再WQLinear.from_linear获取量化后的q_linear之后再替换原有的线性层
    '''
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)#取model中的全部block
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)#拿到单个block中带名字的线性层
        scale_activations(layer)#将单个block中ffn的激活函数变成缩放版本的激活函数

        for name, module in named_linears.items():#此处的module就是对应一个具体的nn.Linear
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)#将layers[i]中的具体的线性层替换成量化版本的线性层，在此之前已经完成了对ffn缩放版激活函数的替换
            else:#如果不是仅初始化
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(#通过伪量化拿到【被离散化到了量化精度对应水平的】权重，缩放因子和零点
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                #此时的scales的形状是(w.shape[0],-1) 如果w的形状是out_feature * in_feature,那么scales的形状就是(out_feature,in_feature/q_group_size) zero也是这个形状
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
