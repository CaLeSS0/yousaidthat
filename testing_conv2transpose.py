
import torch.nn as nn
import torch
import torch.nn.functional as F

# inputs = torch.ones(1, 1, 4, 4)
inputs = torch.tensor([[[[3, 0, 3],
                         [0, 0, 0],
                         [1, 0, 1]]]])
inputs = inputs.type(torch.LongTensor)


weights = torch.tensor([[[[1, 2, 3],
                         [0, 1, 0],
                         [2, 1, 2]]]])

# weights = torch.tensor([[[[2, 1, 2],
#                          [0, 1, 0],
#                          [1, 2, 3]]]])

weights = weights.type(torch.LongTensor)

# inputs = F.pad(inputs, (2, 2, 2, 2))
# print(inputs)

res = F.conv_transpose2d(inputs, weights, stride=(2, 2), padding=1, output_padding=0)
print(res.shape)
print(res)