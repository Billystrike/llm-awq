import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
try:
    from tinychat.models import LlavaLlamaForCausalLM
except ImportError as e:
    pass

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

__all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVL3":
        layers = model.language_model.model.layers
        # layers = [model.language_model.model.layers, model.vision_model.encoder.layers]
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    '''
    把不同结构模型的“嵌入相关层”移动到指定设备上，以保证模型在量化、推理或混合设备分布时不会出错。
    '''
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternVL3":
        model.language_model.model.embed_tokens = (
            model.language_model.model.embed_tokens.to(device)
        )
        model.language_model.model.rotary_emb = (
            model.language_model.model.rotary_emb.to(device)
        )
        model.vision_model.embeddings.to(device)
    elif isinstance(model, LlavaLlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif "llavallamamodel" in str(model.__class__).lower():
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True, #是否自动搜索对应model中的各block中的线性层的最佳缩放因子s（不是零点量化的缩放因子）
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval", # 现在只支持这个校验数据集
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower(): #针对 BigCode 系列模型 设备迁移问题的特定修复，BigCode 模型的注意力掩码 bias张量可能不会随 model.to(device)自动移动
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda") #显式将 model.transformer.bias移动到目标设备

    layers = get_blocks(model)

    samples = get_calib_dataset( #拿到按block_size分割好的校验数据集 形状为[[1,block_size],[1,block_size],...(n_split个)[1,block_size]]
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0) #将校验数据集合并为 ( n_split, block_size )

    inps = [] #用来保存模型第一个 block 的输入（即 input tensor）。
    layer_kwargs = {} #保存该层 forward 的额外参数（例如 attention mask、position ids 等）

    layers[0] = layers[0].cuda() #把第一个block放到cuda上
    move_embed(model, "cuda") #将模型的嵌入相关层放到cuda上

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp) #将输入保存到外部变量
            layer_kwargs.update(kwargs) #同样地，kwargs也保存到外部
            raise ValueError  # early exit to break later inference 故意抛出异常 因为我们只要捕获模型第一层输入就够了，后面的推理没必要进行。

    # patch layer 0 to catch input and kwargs 这一步把模型的第一层临时替换成 Catcher，
    layers[0] = Catcher(layers[0])
    try: #执行模型 forward，直到被 Catcher 中断
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        elif model.__class__.__name__ == "InternVL3":
            model.language_model(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore 还原模型的第一层
    inps = inps[0] #经过第一个block计算过后 sample (n_split,block_size)经过嵌入处理->inp (n_split,block_size,in_feature) 由于inps是一个list所以通过[0]取出输入张量（此时list中也只有这一个元素）

    layers[0] = layers[0].cpu() #现在这两部分暂时不再需要 GPU 计算，所以都移动回 CPU，以节省显存
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i] #拿到第i个block 将其放到cuda上
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict): #钩子函数
            '''
            m：被监控的模块（线性层）
            x：输入元组 (input_tensor,)
            y：输出张量
            name：线性层的名称（如 'attention.q_proj'）
            feat_dict：存储输入值（激活值）的字典
            '''
            x = x[0] #输入的参数是元组，通过x[0]拿到真实的输入张量
            x = x.detach().cpu() #断开与计算图的链接并转移到cpu上减少GPU显存负担
            feat_dict[name].append(x) #向字典中加入对应key 为name 的value x

        input_feat = defaultdict(list) #使用defaultdict(list) 创建字典 当key没有出现的时候会直接创建key 存储对应的value 不会报错。【注意】存储的value是list形式！！
        handles = [] #记录hook的句柄，后面会回收hook
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook( #给每一个线性层注册一个前向传播的钩子
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0] #让整个block进行一次前向传播，计算结果格式为(outputs,*others) 用[0]取出输出张量作为下一层的输入 形状仍为(n_split,block_size,in_feature)
        #第i个block进行前向传播的时候每个线性层都触发了钩子函数，input_feat中每层对应地形状是(n_split,block_size,in_feature) 其中in_feature因具体层而异 例如瓶颈结构的先升后降
        for h in handles:
            h.remove() #移除钩子函数
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()} #v是包含单个元素的list 所以cat操作不影响 k是每个线性层的名字
                                                                            # 经过cat之后字典的value 形状(n_split,block_size,in_feature)

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block( #对block中 注意力机制的KQV线性投影 和最终的多头注意力输出线性投影 以及两个FFN 分别查询最佳的scales
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            #scale_list=[(prev_op_name, [layer_name], scale)] 收录了整个layer的 注意力层、MLP层的scales
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            # 对整个block中的线性层应用缩放
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

            # append prefix to make names global    加上前缀来唯一区分这个scales
            # e.g. (OPTDecoderLayer.self_attn_layer_norm,                          对应prev_op_name
            #      ([OPTDecoderLayer.self_attn.q_proj,                             对应[layer_name]
            #        OPTDecoderLayer.self_attn.k_proj,
            #        OPTDecoderLayer.self_attn.v_proj,]),
            #        scales)                                                        对应scale
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()
        # for line in torch.cuda.memory_summary().splitlines():
        #     if "Allocated" in line:
        #         print(line)

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat, #注意此时的input_feat已经经过apply_scale这一函数缩放了，已经÷scales了
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global 和上面awq_results["scale"]一样的逻辑，加上全局名字
            # e.g. (OPTDecoderLayer.self_attn.v_proj, max_val)
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        # for line in torch.cuda.memory_summary().splitlines():
        #     if "Allocated" in line:
        #         print(line)

    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])
