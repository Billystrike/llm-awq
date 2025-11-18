import torch
import accelerate


def get_module_by_name_suffix(model, module_name: str): #根据后缀返回名称匹配的layer
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    '''
    ①检查是否整模型一个设备 → 是 → .to(device) → return
    ②否则：
    找出 tied 参数
    确定主要执行设备（GPU）
    为 CPU 模块注册 offload hook（按顺序链接）
    为 GPU 模块注册 AlignDevicesHook
    重新连接 tied 参数
    记录并返回模型

    可能的device_map示例：
    device_map = {
    "transformer.wte": "cpu",      # embedding 层放 CPU
    "transformer.blocks.0": "cuda:0",
    "transformer.blocks.1": "cuda:0",
    "lm_head": "cpu",
    }
    '''
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook #AlignDevicesHook：确保模块在执行时输入输出张量都在正确设备上（会自动把输入 tensor .to(device)）

    if "" in device_map:
        d = device_map[""] #如果device_map中有空字符串，将整个模型放在同一个设备上
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model) #找出这些“共享参数对”，后面分配设备时要重新连接（避免丢失共享关系）。
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == { #如果所有层都在 "cpu" 或 "disk" 上（即全在 CPU 或全离线存储），那“主要执行设备”就是 "cpu"；
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else: #否则，就取第一个不是 "cpu" 或 "disk" 的设备作为主执行设备；  所谓主设备就是在把 CPU 层在执行时迁移到主设备上执行。
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"] #找出所有被分配到 "cpu" 的子模块；
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n) #根据device中的映射关系 取到name对应的layer
        _, prev_hook = accelerate.cpu_offload_with_hook(#把该层的参数暂时留在CPU，当执行到该层的前向传播时自动从CPU把权重加载到 execution_device执行计算后把权重释放回 CPU
            m, execution_device=main_device, prev_module_hook=prev_hook #prev_module_hook：链接前一个 hook，这样可以串成链（防止乱序 offload）。
                                                              #prev_hook 记录上一个模块的 hook，用于下一个模块连接
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]
        )._hf_hook.prev_module_hook = prev_hook #让第一个 CPU 模块的 hook 的前驱是最后一个 hook，让整个 offload 流程连成一圈，形成完整的依赖链，保证计算顺序正确

    for n, d in device_map.items(): #遍历所有模块
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":  #如果它不是 CPU 层：
            d = torch.device(d) #转成 torch.device；
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True) #创建AlignDevicesHook确保输入输出tensor都在d。place_submodules=True子模块也放到同设备；
            add_hook_to_module(m, hook) #把这个 hook 附加到该模块上。
    accelerate.utils.modeling.retie_parameters(model, tied_params) #前面在分设备时打断了模块之间的参数共享；重新让共享的权重指针一致（确保 embedding/lm_head 等仍然共享）。
    model.hf_device_map = device_map #把 device_map 记录在模型属性里

    return model
