from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()#命令行参数解析器，自动把命令行参数转为Python变量
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config                                  #action="store_true" 当命令行中出现--parallel表示此参数设置为True
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)#量化位数
parser.add_argument("--q_group_size", type=int, default=-1)#量化group分组大小
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")#是否使用零点，默认使用
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])#fake表示假量化，仅模拟量化。先quant后再dequant，模拟量化造成的精度损失
# save/load real quantized weights                                                                  其权重数据类型仍是高精度，用于测试量化效果。
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument(
    "--dump_fake", type=str, default=None, help="save fake-quantized model" #保存伪量化模型
)
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results" #给定具体的存放AWQ搜索结果存放路径
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",#一个专用开关，用于特定模型（VILA 1.5）。
    action="store_true",
    help="quantizing vila 1.5",
)
parser.add_argument(
    "--vila-20",#同上，用于特定模型
    action="store_true",
    help="quantizing or smoothing vila 2.0 (NVILA)",
)
parser.add_argument(
    "--smooth_scale",#打开视觉塔激活分析功能；生成视觉塔的激活缩放比例（act scale）
    action="store_true",
    help="generate the act scale of visiontower",
)
parser.add_argument(
    "--media_path",#输入视频路径，这些视频会被模型的视觉塔用来分析激活分布，从而计算出激活的 scale 值。
    type=str,
    nargs="+",
    help="The input video to get act scale for visiontower",
)
parser.add_argument(
    "--act_scale_path",#计算出来的激活 scale 要保存到哪个路径。
    type=str,
    default=None,
    help="Path to save act scale",
)
args = parser.parse_args()#将输入命令行的参数分割的结果赋值给args
assert ( #如果你启用了 --smooth_scale，那么必须同时提供--act_scale_path（要保存的文件路径）、--media_path（至少一个视频路径）
    args.act_scale_path is not None and len(args.media_path) > 0
) or not args.smooth_scale
vila_10_quant_mode = (#判断当前模型属于哪个版本的多模态模型
    ("llava" in args.model_path.lower() or "vila" in args.model_path.lower()) #如果模型路径中出现了'llava'或'vila'
    and not args.vila_15 #并且用户没有手动指定--vila-15
    and not args.vila_20
)
#Example: 0:10GiB 1:10GiB cpu:30GiB
max_memory = [v.split(":") for v in (args.max_memory or [])] #根据冒号来分割模型并行情况下各显卡分匹配的显存量。
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}#根据分割的结果构建字典

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path, dtype):
    '''
    这个函数你可以
    1 args.load_quant读取以往已经量化成int4的权重来获取量化后的模型
    2 如果没有现成已经量化成int4的模型权重可以读取，那就现场跑一遍量化流程
    2.1.1 args.load_awq 如果有之前计算过的AWQ搜索结果那就直接读取搜索结果对权重进行scales和clip。
    2.1.2 args.run_awq 反之如果没有进行过AWQsearch的话还要再跑一边AWQsearch，注意AWQsearch执行完之后会直接【exit】不会自动apply 需要在运行一遍走2.1.1
    2.2应用完scales和clip之后开始执行权重量化
    2.2.1 args.q_backend == "fake" 执行伪量化，先量化紧接着反量化，维持FP精度，目的是为了检验AWQ量化的效果
    2.2.2 args.q_backend == "real" 执行真正的量化流程：先伪量化计算scales和zero，再传给WQLinear实例化weight quantized linear 将model 的权重真的变成INT
    2.3执行完real quantization之后通过args.dump_quant指定保存路径 将量化之后的模型权重保存，下次就可以通过1读取了。
    最终返回的结果是经过量化之后的 model 和 对应model的enc（或者说tokenizer）
    '''
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False},
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)# 根据model_path加载模型的配置类 trust_remote_code=True允许从模型仓库中加载自定义 Python 代码
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2； 4.36.2 版本后默认会在推理时缓存中间状态（key, value）
        config.use_cache = False#因为量化过程要对大量层进行扫描和统计激活分布 缓存这些内容没有意义，还会导致 GPU 显存爆炸。所以这里显式关闭。
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config( #from_config 只是加载模型的结构，权重随机
                config=config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        real_quantize_model_weight( #由于init_only 所以将model中的全部带名字的线性层替换成初始化的q_linear，权重矩阵为0矩阵
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights() #将模型的嵌入层和输出层权重绑定

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {} #如果之前显示指定了每个GPU的显存分配，就将他变为字典中的值，方便传给infer_auto_device_map
        device_map = infer_auto_device_map( #根据device map优先将模型放在GPU上，如果GPU显存不够再放到CPU上，CPU内存不够再放到硬盘上
            model,
            no_split_module_classes=[ #不能跨设备的块，这些块含有残差连接，如果跨设备计算来回传输数据消耗时间过多。
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant, #load_quant: 量化权重文件路径
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized 如果不加载已有的量化权重那就现场计算量化
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq 现场计算量化又有两个分支
        #  ①现场进行AWQ搜索来查询对于model而言每个block每个fc权重最佳的scales和clip，此时只进行搜索，搜索完成就结束②读取以往已经计算过的AWQ搜索结果直接应用
        # Init model on CPU:
        kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained( #from_pretrained 会加载完整的权重
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        if args.run_awq: #参数中设置了执行AWQ搜索的话
            assert args.dump_awq, "Please save the awq results with --dump_awq" #需要指定将搜索结果放置的路径
            from awq.quantize.pre_quant import run_awq
            awq_results = run_awq( #执行AWQ搜索 在搜索的过程中同时对model的fc权重应用了scales & clip
                model,
                enc, #模型的tokenizer
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

            exit(0) #执行完AWQ搜索就退出

        if args.load_awq: #读取并应用预先计算完成的AWQ搜索结果，为模型权重进行scale和clip 之后再当场计算权重的零点量化，打包成硬件友好的格式。
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None #如果是fake量化就不要保存量化结果了
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization 执行真正的量化流程：先伪量化计算scales和zero，再传给WQLinear实例化weight quantized linear 。
                # 再按照group实际针对w每一个in_feature进行零点量化，将得到的结果按照内存友好的形式打包，按照WQLinear的格式存放各量化参数。最后将model中每个block的每个linear 替换成wqlinear
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt") #自动将保存文件名后缀改为-v2.pt
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    # a hack here to auto set model group  此处针对多模态大模型
    if args.smooth_scale and args.vila_20:
        if os.path.exists(args.act_scale_path):
            print(f"Found existing Smooth Scales {args.act_scale_path}, skip.")
        else:
            from awq.quantize import get_smooth_scale

            act_scale = get_smooth_scale(args.model_path, args.media_path)
            os.makedirs(os.path.dirname(args.act_scale_path), exist_ok=True)
            torch.save(act_scale, args.act_scale_path)
            print("Save act scales at " + str(args.act_scale_path))
            args.model_path = args.model_path + "/llm"
        if args.dump_awq is None and args.dump_quant is None:
            exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()
    model, enc = build_model_and_enc(args.model_path, args.dtype)

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = 2048
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            task_names = args.tasks.split(",")

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
    #测试一下，我将代码IDE换成了VScode，现在看看github是否正常工作，一会再看一下是否可以远程连接到autoDL，接下来再读一下代码，后面就打算开始修改代码了
