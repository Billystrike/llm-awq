import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


def make_divisible(c, divisor):
    '''
    向上取整的除法，确保结果是 divisor 的倍数,e.g. c=10 divisor=8 return 2
    '''
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    '''
    计算零点（zero-point）参数在内存中的对齐宽度，以适配 GPU 批量访问需求。
    
    核心思路：
    1. 分组量化后，共有 (in_features // group_size) 个 group，每个 group 需要 1 个零点参数
    2. GPU 不希望逐个读取零点，而是按 pack_num 为单位批量加载（如一次加载 8 个 group 的零点）
    3. 因此需要将 group 数量向上对齐到 pack_num 的倍数
    4. 针对小 group_size（32/64），零点更密集，需要更高的 size_multiplier 进行二次对齐
    
    参数：
        in_features: 输入特征维度（权重矩阵的列数）
        group_size: 分组量化的组大小，默认 128
        pack_num: 批量打包单位，默认 8（即每次加载 8 个 group 的量化参数）
    
    返回：
        对齐后的宽度（需要多少个 pack_num 单元来存储所有零点参数）
    
    示例：
        in_features=4096, group_size=128 → 32 个 group → 32/8=4 个 pack → 返回 4
        in_features=1024, group_size=128 → 8 个 group → 8/8=1 个 pack → 返回 1
    '''
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:#如果组size较小 就乘一个大一些的打包密度因子，用来对其硬件
        size_multiplier = 2
    elif group_size == 32:#组越小零点越多 密度因子就越大
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)#先计算组的数量，再将其除以Pack_num，向上取整。计算需要多少个pack_num大小的块来存放量化参数。
    #假设in_features = 4096 group_size = 128 会有 32 个 group；给这些 group 分配相应的量化参数（scale, zero-point）。这些参数会被 以 pack_num 为单位 打包存放在内存中
    #这就相当于“我们不希望 GPU 一次只加载 1 个 group 的量化参数，而是一次性加载 pack_num（8） 个 group 的量化参数，提升并行度”。于是我们就得保证：group 的数量能被 pack_num 整除。
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier#针对不同 group_size二次 padding
    # 再把 base_width 调整成 size_multiplier 的倍数，以保证 GPU 访存对齐和 warp 并行时的正确布局。
    return base_width


def pack_intweight(unpacked_qweight, interleave, kstride):
    '''
    将未打包的 4-bit 整数权重矩阵重排并打包为 GPU 友好的内存布局（int16 格式）。
    
    核心目标：
    1. 位交错重排（bit interleaving）：优化 GPU 向量化解包效率
    2. 行交错存储（row interleaving）：提升 GPU warp 并行度
    3. 打包压缩：将 4 个 4-bit 值合并成 1 个 int16，节省显存
    
    处理流程（三个关键步骤）：
    【步骤 1】将每 32 个连续元素重排为 4×4×2 交错模式
        - 原始顺序：[0,1,2,...,31]
        - 交错后：[0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,...]
        - 目的：使 GPU SIMD 指令能高效解包
    
    【步骤 2】将每 8 个元素按偶奇分离重排
        - 原始顺序：[0,1,2,3,4,5,6,7]
        - 重排后：[0,2,4,6,1,3,5,7]
        - 目的：匹配 GPU 反量化 kernel 的计算模式，减少分支
    
    【步骤 3】行交错 + 打包为 int16
        - 将每 interleave(4) 行交错存储，提升并行度
        - 将 4 个 4-bit 值通过位运算合并成 1 个 int16
          公式：packed = val0 | (val1<<4) | (val2<<8) | (val3<<12)
    
    参数：
        unpacked_qweight: 未打包的量化权重张量 [N, K]，元素为 4-bit 整数（存储为 int32）
        interleave: 行交错数量，默认 4（每 4 行作为一组交错存储）
        kstride: K 维度的步长，默认 64（控制交错粒度）
    
    返回：
        打包后的权重张量 [N//interleave, K]，dtype=torch.int16
    
    注意：此函数逻辑复杂，主要服务于 AWQ CUDA kernel 的特定内存布局需求
    '''
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0] #out_feature
    K = unpacked_qweight.shape[1] #in_feature

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...] transpose之后形成了四路交错的形式
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)#把原本连续的 32 个位置组织成符合硬件向量化解包更高效的顺序
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)#恢复原本的形状，但此时内部元素顺序已经变得交错了

    # reorder each 8 weights for fast dequantization
    # 【为什么还要二次重排序？】为了在 GPU 解包并反量化时能用矢量/分支少的指令快速把 8 个 4-bit 值转换成 8 个浮点值。换句话说，这是数据布局对 compute pattern 的优化。
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)#元素已经被按解包友好的次序排列

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
        Packed_Kernel[..., 0]#取最后一个维度的第0个列，前面的维度全要
        | (Packed_Kernel[..., 1] << 4)#左移四位进行或操作
        | (Packed_Kernel[..., 2] << 8)#左移四位进行或操作
        | (Packed_Kernel[..., 3] << 12)#左移四位进行或操作
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight


class ScaledActivation(nn.Module):
    '''
    带缩放因子的激活函数模块。将原本的激活函数输出除以一个可学习的缩放因子，从而调整激活值的分布。
    这样做的目的是为了配合量化后的线性层，使得激活值的范围更适合量化表示，减小量化误差对模型性能的影响。
    '''
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)#将输入的缩放因子 scales转换为 PyTorch 的可训练参数。保留原始缩放因子的数值（通过 .data获取），但断开与原计算图的连接

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)#self.scales.view(1, 1, -1) 将一维的缩放因子张量重塑（reshape）为三维张量适配不同维度的输入张量


class WQLinear(nn.Module):
    '''
    量化权重的线性层，支持4-bit权重量化，采用分组量化和内存交错优化以提升GPU计算效率。
    '''
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, dtype=torch.float16):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8#Split-K 并行拆分 in_features（K 维度） 被分成 8 份 独立计算，结果 sum 归约
        self.interleave = 4 #4 路内存交错 将权重每4个输出通道打包
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0 #确保in_feature能够被group_size整除，采用分组量化，权重形状为out_feature * in_feature。量化是按行方向从左到右进行的
        assert out_features % (32 // self.w_bit) == 0 #打包却是按照列方向从上到下进行的，我知道这有些反直觉，但是Y=XW^T W经过转置之后就正常了【待进一步确认】
        pack_num = 32 // self.w_bit #32bit //4bit = 8 每个32int整数能放8个 4bit权重
        int16_pack_num = 16 // self.w_bit #若使用int 16 则每个整数能存放 4个 4bit权重

        assert out_features % (self.interleave) == 0 #一次加载权重块时 同时读到 多个输出通道的连续权重；
        self.register_buffer( #注册一个不会参与训练（没有梯度）的张量 qweight. 不会更新，只用于前向推理。
            "qweight",
            torch.zeros(
                (
                    out_features // self.interleave, #交错存储压缩后的输出通道数
                    in_features // int16_pack_num * self.interleave, #in_features // int16_pack_num：每 4 个权重打包成 1 个 int16
                                                                     #打包后的长度再乘以interleave，表示我们把这 4 个通道的权重交错展开。
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num, #正常情况下形状(n_group,out_feature)
                    out_features,
                ),
                dtype=dtype,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=dtype,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=dtype, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear( # 将现成的线性层转变成量化版线性层：将权重进行量化并打包成硬件友好的形状，同时返回缩放因子和缩放般的零点。如果只是初始化就返回一个全零的矩阵壳子
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        '''
        将一个普通的线性层转换为量化权重的线性层，并进行必要的量化和打包处理。
        返回一个 WQLinear 实例，包含量化后的权重、缩放因子和零点等信息，如果 init_only 为 True，则仅返回初始化的量化层结构，权重和参数均为零。
        参数：
            linear: 要转换的普通线性层（nn.Linear 实例）
            w_bit: 量化位数（目前仅支持 4-bit）
            group_size: 分组量化的组大小
            init_only: 是否仅初始化量化层而不进行实际量化（默认为 False）
            scales: 预计算的缩放因子（可选）
            zeros: 预计算的零点（可选）
        '''
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            dtype=linear.weight.data.dtype
        )
        if init_only:  # just prepare for loading sd 返回的全是0矩阵，只是形状符合要求，具体的值还没填
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales #对应的形状(out_feature,in_feature/q_group_size)=(out_feature,n_group)

        dtype = scales.dtype

        pack_num = 32 // awq_linear.w_bit #8 因为现在只支持4bit quantization
        qscales = torch.zeros(
            (
                scales.shape[0], #out_feature
                calculate_zeros_width(linear.in_features, group_size) * pack_num,#为什么不直接设置为scales.shape[1]？为了内存对齐、向量化读取以及符合后端 kernel 的访问模式
            ), #在对其的情况下计算结果形状就是 (out_feature,n_group)
            dtype=dtype,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales #正常情况下形状维持(out_feature,n_group).将真实的 scale 数据放在左边，右侧是 padding（零），以满足对齐要求 。qscales的形状(out_feature,aligned_width)。
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales.transpose(1, 0).contiguous()#为何transpose？因为上面init类定义的时候就是要求(n_group,out_feature)这个形状，刚好相反
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().to(dtype)

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) #此处变量的形状都是(out_feature,) 此处对应地是量化操作
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None] # 将量化值变成整数，加上[:, None]是为了让变量的形状升一个维度变成(out_feature,1)方便一会进行cat操作
            )
        intweight = torch.cat(intweight, dim=1) #沿着刚加的维度进行cat操作，结果形状(out_feature，in_feature)对应有out_feature行，每行有in_feature个元素。全部转化为了int型
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)# 当前的intweight是“未打包”的整数量化值矩阵
        awq_linear.qweight = pack_intweight( # 将打包之后的权重赋给awq_linear的注册属性qweight
            intweight.contiguous(), interleave=4, kstride=64
        )

        zeros = zeros.to(dtype=torch.int32)
        scaled_zeros = torch.zeros_like(qscales)# out_feature,n_group 一个空矩阵等待赋值
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32)) #很像scale_zeros 的操作，都是零值乘缩放因子，此处只是将缩放零值的形状变成(out_feature,n_group)
        ).to(dtype)
        awq_linear.scaled_zeros = scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        if inputs.numel() / inputs.shape[-1] < 8:
            out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_inference_engine.gemm_forward_cuda_new(
                inputs, self.qweight, self.scales, self.scaled_zeros
            )  # - 8.0 * self.scales)
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
