import numpy as np
import pickle
import argparse
from os.path import exists, isdir, basename, join, splitext
from sklearn.model_selection import train_test_split
from extract_features import get_folders, get_files

from scripts.custom_dataset import TalkingFaceDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from functools import reduce
from operator import __add__

import matplotlib.pyplot as plt
import tqdm

class TalkingHeadModel(nn.Module):
	def __init__(self, num_still_images):
		super(TalkingHeadModel, self).__init__()
		self.num_still_images = num_still_images
		self.dim = self.num_still_images*3 + 3

		self.x_skip1 = None
		self.x_skip2 = None
		self.x_skip3 = None

		self.audio_encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same'),
			nn.BatchNorm2d(64, momentum=0.8),
			nn.ReLU(inplace=True),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
			nn.BatchNorm2d(128, momentum=0.8),
			nn.ReLU(inplace=True),

			nn.MaxPool2d((3, 3), stride = (1, 2), padding=1),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
			nn.BatchNorm2d(256, momentum=0.8),
			nn.ReLU(inplace=True),

			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
			nn.BatchNorm2d(256, momentum=0.8),
			nn.ReLU(inplace=True),

			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
			nn.BatchNorm2d(512, momentum=0.8),
			nn.ReLU(inplace=True),

			nn.MaxPool2d((3, 3), stride = 2, padding=1),
			nn.Flatten(),
			nn.Linear(27648, 512),
			nn.ReLU(inplace=True),

			nn.Linear(512, 256),
			nn.ReLU(inplace=True)
		)

		# Identity encoder
		self.conv1 = nn.Conv2d(in_channels=self.dim, out_channels=96, kernel_size=7, stride=2, padding=3)
		self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=2)
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
		self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')
		self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')

		self.batchnorm1 = nn.BatchNorm2d(96, momentum=0.8)
		self.batchnorm2 = nn.BatchNorm2d(256, momentum=0.8)
		self.batchnorm3 = nn.BatchNorm2d(512, momentum=0.8)
		self.batchnorm4= nn.BatchNorm2d(512, momentum=0.8)
		self.batchnorm5 = nn.BatchNorm2d(512, momentum=0.8)

		self.relu = nn.ReLU(inplace=True)
	
		self.maxpool = nn.MaxPool2d((3, 3), stride = 2, padding=1)

		self.flatten = nn.Flatten()
		self.dense1 = nn.Linear(25088, 512)
		self.dense2 = nn.Linear(512, 256)

		# Decoder part
		self.dense3 = nn.Linear(512, 98)
		
		self.convtranspose_p_same1 = nn.ConvTranspose2d(2, 512, kernel_size=6, stride=1, padding=5)
		self.convtranspose_p_same2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1, padding=4)

		self.convtranspose_up1 = nn.ConvTranspose2d(512, 96, 4, stride=2, padding=1)
		self.convtranspose_up2 = nn.ConvTranspose2d(352, 96, 4, stride=2, padding=1)
		self.convtranspose_up3 = nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1)
		self.singleconvtranspose = nn.ConvTranspose2d(64, 3, (4, 4), stride=2, padding=1)
		self.sigmoid = nn.Sigmoid()

		self.batchnorm_transpose1 = nn.BatchNorm2d(512, momentum=0.8)
		self.batchnorm_transpose2 = nn.BatchNorm2d(256, momentum=0.8)
		self.batchnorm_transpose3 = nn.BatchNorm2d(96, momentum=0.8)
		self.batchnorm_transpose4 = nn.BatchNorm2d(96, momentum=0.8)
		self.batchnorm_transpose5 = nn.BatchNorm2d(64, momentum=0.8)



	def convolution(self, x, in_, out_, kernel_size=3, strides=1, padding='same'):
		x = nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=kernel_size, stride=strides, padding=padding)(x)
		x = nn.BatchNorm2d(out_, momentum=0.8)(x)
		x = nn.ReLU(inplace=True)(x)
		return x

	def transposed_convolution(self, x, in_, out_, kernel_size=3, strides=1): # 14, 14, 96
		pad = self.calculate_padding((kernel_size, kernel_size))
		x = F.pad(x, pad)
		x = nn.ConvTranspose2d(in_, out_, kernel_size=kernel_size, stride=strides, padding=kernel_size-1)(x)

		x = nn.BatchNorm2d(out_, momentum=0.8)(x)
		x = nn.ReLU(inplace=True)(x)
		return x

	def upsample(self, x, in_, out_, kernel_size, strides):
		upsample = nn.ConvTranspose2d(in_, out_, kernel_size, stride=strides, padding=1)(x)
		x = nn.BatchNorm2d(out_, momentum=0.8)(upsample)
		x = nn.ReLU(inplace=True)(x)
		return upsample

	def calculate_padding(self, kernel):
		conv_padding = reduce(__add__, 
			[(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel[::-1]])
		
		return conv_padding
	
	def identity_encoder_net(self, identity):
		# Convolution layer 1
		x = self.conv1(identity)
		x = self.batchnorm1(x)
		x = self.relu(x)

		# Max pooling layer 1
		self.x_skip1 = self.maxpool(x)

		# Convolution layer 2
		x = self.conv2(self.x_skip1)
		x = self.batchnorm2(x)
		self.x_skip2 = self.relu(x)

		# Max pooling layer 2
		self.x_skip3 = self.maxpool(self.x_skip2)

		# Convolution layer 3
		x = self.conv3(self.x_skip3)
		x = self.batchnorm3(x)
		x = self.relu(x)

		# Convolution layer 4
		x = self.conv4(x)
		x = self.batchnorm4(x)
		x = self.relu(x)

		# Convolution layer 5
		x = self.conv5(x)
		x = self.batchnorm5(x)
		x = self.relu(x)

		# Dense layer 1
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.relu(x)

		# Output layer
		encoded_identity = self.dense2(x)
		encoded_identity = self.relu(encoded_identity)
		return encoded_identity

	def decoder_layer(self, concatenated_features):
		x = self.dense3(concatenated_features)
		x = self.relu(x)
		x = x.view(-1, 2, 7, 7)

		# Transpose layer 1
		pad = self.calculate_padding((6, 6))
		x = F.pad(x, pad)
		x = self.convtranspose_p_same1(x)
		x = self.batchnorm_transpose1(x)
		x = self.relu(x)

		# Transpose layer 2
		pad = self.calculate_padding((5, 5))
		x = F.pad(x, pad)
		x = self.convtranspose_p_same2(x)
		x = self.batchnorm_transpose2(x)
		x = self.relu(x)

		# Concat layer 1
		x = torch.cat([x, self.x_skip3], 1)

		# Upsampling layer 1
		x = self.convtranspose_up1(x)

		# Concat layer 2
		x = torch.cat([x, self.x_skip2], 1)

		# Upsampling layer 2
		x = self.convtranspose_up2(x)

		# Concat layer 3
		x = torch.cat([x, self.x_skip1], 1)

		# Upsampling layer 3
		x = self.convtranspose_up3(x)

		# Single upsampling layer
		decoded = self.singleconvtranspose(x)
		decoded = self.sigmoid(decoded)

		return decoded

	def forward(self, audio, identity):
		encoded_audio = self.audio_encoder(audio)
		encoded_identity = self.identity_encoder_net(identity)

		concat = torch.cat([encoded_audio, encoded_identity], 1)

		decoded = self.decoder_layer(concat)
		return decoded

def model(files, batch_size, epochs, steps_per_epoch, gpus, num_still_images, model_dir):
	train_files, val_files = train_test_split(np.array(files), test_size=0.1)
	train_files = train_files.astype(object)
	val_files = val_files.astype(object)

	training_generator = TalkingFaceDataset(train_files, num_still_images, batch_size)

	# validation_generator = TalkingFaceDataset(val_files, num_still_images, batch_size)

	lr = 1e-3
	epochs = 1

	model = TalkingHeadModel(num_still_images)
	optimizer = optim.Adam(model.parameters(), lr)
	criterion = nn.L1Loss()

	for epoch in range(epochs):
		running_loss = 0
		
		pbar = tqdm.tqdm(training_generator)
		i_batch = 0
		for (X_audio, X_identity, Y) in pbar:
			if X_audio == []:
				break

			optimizer.zero_grad()
			outputs = model(X_audio, X_identity)
			loss = criterion(outputs, Y)
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
		
			if i_batch % 10 == 9:
				pbar.set_description(f"Epoch: {epoch+1}/{epochs} batch: {i_batch+1}/{len(training_generator)} | loss: {running_loss/i_batch:.3f}")

			i_batch += 1

		PATH = 'saved_models/best_model_pytorch.pth'
		torch.save(model.state_dict(), PATH)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-f', '--feature_path', help='Feature path (path containing the saved audio \
						and frames', default="features/")
	parser.add_argument('-b', '--batch_size', default=64, required=False, \
						help='Batch size to train the model')
	parser.add_argument('-e', '--epochs', default=20, required=False, \
						help='No of epochs to train the model')
	parser.add_argument('-spe', '--steps_per_epoch', default=1000, required=False, \
						help='No of steps per epoch')
	parser.add_argument('-g', '--no_of_gpus', default=0, required=False, help='No of GPUs')
	parser.add_argument('-s', '--no_of_still_images', default=1, required=False, \
						help='No of still images')
	parser.add_argument('-md', '--model_directory', default='saved_models/', \
						help='Path to save the model')

	args = parser.parse_args()


	print("---------------------")
	feature_path = args.feature_path
	folders = get_folders(feature_path)
	num_folders = len(folders)
	print("Total number of folders = ", num_folders)


	files = []
	for folder in folders:
		sub_folder_path = join(feature_path, folder)
		sub_folders = get_folders(sub_folder_path)
		for sub_folder in sub_folders:
			file_path = join(sub_folder_path, sub_folder)
			file = get_files(file_path, extension=['.jpg'])
			files.extend(file)

		
	model(files, int(args.batch_size), int(args.epochs), int(args.steps_per_epoch), int(args.no_of_gpus), \
			int(args.no_of_still_images), args.model_directory)

	