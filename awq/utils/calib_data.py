import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512): #准备量化校准数据集
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation") #Pile validation 集合的备份版本，专门用于量化实验的验证。
    else:#传别的直接报错
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"] #取出一条文本。
        line = line.strip() #除首尾空格。
        line_encoded = tokenizer.encode(line) #结果是一个长为sequence_length的list
        if len(line_encoded) > 512: #如果encoded之后的数据太长就跳过
            continue
        sample = torch.tensor([line_encoded]) #每个sample的形状都是 [1, sequence_length]
        if sample.numel() == 0: #若sample为空就跳过
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)  #拼接后 cat_samples 的形状是：[1, sum(sequence_length)] :一个大的 token 序列，长度为所有样本 token 的总数。
    n_split = cat_samples.shape[1] // block_size #它把整个拼接的 token 流均匀地划分成若干块：
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
