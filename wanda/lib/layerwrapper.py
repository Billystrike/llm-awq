import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0] #output dim
        self.columns = layer.weight.data.shape[1] #input dim

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        '''
        函数的目标是收集输入数据的统计信息，以便后续计算 Wanda 权重时使用。
        scaler_row: 用于累积输入数据的平方和的张量，形状为 [input_dim]。
        '''
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        #tmp 记录当前批次的样本数量
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1])) #batch_size*seq_len, hidden_size
            inp = inp.t() #转置以便后续计算

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples