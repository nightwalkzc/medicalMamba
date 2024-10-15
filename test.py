import torch
from block.gcn_lib.torch_vertex import Grapher
from models.vmunet import decoers

#---------------调试展示Tensor维度
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

from torch import nn

if __name__ == '__main__':


    block = Grapher(768) #输入 C

    # block = decoers(768)

    input = torch.randn(1, 768, 7, 7) # 输入 B C H W

    # Print input shape
    print(input.size())

    # Forward pass through the SHSA module
    output = block(input)

    print(output.size())