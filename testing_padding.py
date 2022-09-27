from keras.layers import Input, Conv2DTranspose, BatchNormalization, Activation

import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import reduce
from operator import __add__

def transposed_convolution(x, filters, kernel_size=3, strides=1, padding='same'):
	x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, \
		padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)
	x = Activation('relu')(x)
	return x

# Audio encoder
input_tf = Input(shape=(7, 7, 2)) 
x_tf = transposed_convolution(input_tf, 512, 6)
print(f"SOURCE | Starting shape: {input_tf.shape} | End shape: {x_tf.shape}")



def transposed_convolution2(x, in_, out_, kernel_size=3, strides=1, padding='same'):
    x = nn.ConvTranspose2d(in_, out_, kernel_size=kernel_size, stride=strides, padding=padding)(x)
    x = nn.BatchNorm2d(out_, momentum=0.8)(x)
    x = nn.ReLU(inplace=True)(x)
    return x

input_pt = torch.rand((1111, 2, 7, 7))
# PADDING SAME NEEDS TO BE HERE, BUT CANNOT MAKE IT WORK

kernel_sizes = (6, 6)
conv_padding = reduce(__add__, 
    [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])
    
x = F.pad(input_pt, conv_padding, )

x_pt = transposed_convolution2(x, 
                                2, 512, 
                                kernel_size=kernel_sizes, 
                                padding=5)

print(f"TARGET | Starting shape: {input_pt.detach().numpy().shape} | End shape: {x_pt.detach().numpy().shape}")
