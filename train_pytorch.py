import numpy as np
import pickle
import argparse
from os.path import exists, isdir, basename, join, splitext
from sklearn.model_selection import train_test_split
from extract_features import get_folders, get_files
# from scripts.multi_gpu_model import *
from datetime import datetime

from scripts.custom_dataset import TalkingFaceDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from functools import reduce
from operator import __add__

# def convolution(x, filters, kernel_size=3, strides=1, padding='same'):
# 	x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
# 	x = BatchNormalization(momentum=.8)(x)
# 	x = Activation('relu')(x)
# 	return x

# def transposed_convolution(x, filters, kernel_size=3, strides=1, padding='same'):
# 	x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, \
# 		padding=padding)(x)
# 	x = BatchNormalization(momentum=.8)(x)
# 	x = Activation('relu')(x)
# 	return x

class TalkingHeadModel(nn.Module):
	def __init__(self, num_still_images):
		super(TalkingHeadModel, self).__init__()
		self.num_still_images = num_still_images

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

	def transposed_convolution_from_F(self, x, in_, out_, kernel_size=3, strides=1): # 14, 14, 96
		x = F.conv_transpose2d(in_, out_, kernel_size=kernel_size, stride=strides, padding='same')(x)

		x = nn.BatchNorm2d(out_, momentum=0.8)(x)
		x = nn.ReLU(inplace=True)(x)
		return x

	def upsample(self, x, in_, out_, kernel_size, strides):
		upsample = nn.ConvTranspose2d(in_, out_, kernel_size, stride=strides, padding=kernel_size-2)(x)
		return upsample

	def calculate_padding(self, kernel):
		conv_padding = reduce(__add__, 
			[(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel[::-1]])
		
		return conv_padding


	def audio_encoder_net(self, audio):
		x = self.convolution(audio, 1, 64, 3) # 64, 12, 35
		x = self.convolution(x, 64, 128, 3)  # 128, 12, 35
		x = nn.MaxPool2d((3, 3), stride = (1, 2), padding=1)(x) # 128, 12, 18

		x = self.convolution(x, 128, 256, 3) # 256, 12, 18
		x = self.convolution(x, 256, 256, 3) # 256, 12, 18
		x = self.convolution(x, 256, 512, 3) # 512, 12, 18

		# x = F.pad(x, (0, 0, 2, 1))
		x = nn.MaxPool2d((3, 3), stride = 2, padding=1)(x) # 512, 6, 9
		x = nn.Flatten()(x) # 27648
		x = nn.Linear(27648, 512)(x) # 512
		x = nn.ReLU(inplace=True)(x) #

		encoded_audio = nn.Linear(512, 256)(x) # 256
		encoded_audio = nn.ReLU(inplace=True)(encoded_audio)
		return encoded_audio
	
	def identity_encoder_net(self, identity):
		dim = self.num_still_images*3 + 3
		x = self.convolution(identity, dim, 96, 7, 2, padding=3) # 96, 56, 56

		x_skip1 = nn.MaxPool2d((3, 3), stride = 2, padding=1)(x) # 96, 28, 28
		x_skip2 = self.convolution(x_skip1, 96, 256, 5, 2, padding=2) # 256, 14, 14
		x_skip3 = nn.MaxPool2d((3, 3), stride = 2, padding=1)(x_skip2) # 256, 7, 7

		x = self.convolution(x_skip3, 256, 512, 3) # 512, 7, 7
		x = self.convolution(x, 512, 512, 3) # 512, 7, 7
		x = self.convolution(x, 512, 512, 3) # 512, 7, 7

		x = nn.Flatten()(x) # 27648
		x = nn.Linear(25088, 512)(x) # 512
		x = nn.ReLU(inplace=True)(x) #

		encoded_identity = nn.Linear(512, 256)(x) # 256
		encoded_identity = nn.ReLU(inplace=True)(encoded_identity)
		return encoded_identity, x_skip1, x_skip2, x_skip3

	def forward(self, audio, identity):
		encoded_audio = self.audio_encoder_net(audio)
		encoded_identity, x_skip1, x_skip2, x_skip3 = self.identity_encoder_net(identity)

		concat = torch.cat([encoded_audio, encoded_identity], 1)

		x = nn.Linear(512, 98)(concat)
		x = nn.ReLU(inplace=True)(x)
		x = x.view(-1, 2, 7, 7)
		
		# # PADDING SAME NEEDS TO BE HERE, BUT CANNOT MAKE IT WORK
		# x = self.transposed_convolution_from_F(x, 2, 512, 6) # 512, 7, 7
		# x = self.transposed_convolution_from_F(x, 512, 256, 5) # 256, 7, 7
		# x = torch.cat([x, x_skip3], 1) # 512, 7, 7
		# x = self.transposed_convolution_from_F(x, 512, 96, 5, 2) # 256, 7, 7

	# x = transposed_convolution(x, 512, 6) # 7, 7, 512
	# x = transposed_convolution(x, 256, 5) # 7, 7, 256
	# x = concatenate([x, x_skip3]) # 7, 7, 512
	# x = transposed_convolution(x, 96, 5, 2) # 14, 14, 96
	# x = concatenate([x, x_skip2]) # 14, 14, 352
	# x = transposed_convolution(x, 96, 5, 2) # 28, 28, 96
	# x = concatenate([x, x_skip1]) # 28, 28, 192
	# x = transposed_convolution(x, 64, 5, 2) # 56, 56, 64
	# decoded = Conv2DTranspose(3, (5, 5), strides=2, activation='sigmoid', padding='same')(x) # 112, 112, 3
	pass

def model(files, batch_size, epochs, steps_per_epoch, gpus, num_still_images, model_dir):
	train_files, val_files = train_test_split(np.array(files), test_size=0.1)
	train_files = train_files.astype(object)
	val_files = val_files.astype(object)

	training_generator = TalkingFaceDataset(train_files, num_still_images, batch_size)
	validation_generator = TalkingFaceDataset(val_files, num_still_images, batch_size)

	t = 0

	model = TalkingHeadModel(num_still_images)
	for param in model.parameters():
		print(param)

	# test = torch.ones((1, 1, 12, 35))
	# model(test, test)

	# print(len(training_generator))
	# for i_batch, (X_audio, X_identity, Y) in enumerate(training_generator):		
	# 	print(i_batch, X_audio.shape)

	# 	model(X_audio, X_identity)


	# for i_batch, image in enumerate(dataloader):
		# print(image[1])

	# Audio encoder
	# input_audio = Input(shape=(12, 35, 1)) 

	# x = convolution(input_audio, 64, 3)
	# x = convolution(x, 128, 3)
	# x = MaxPooling2D((3, 3), strides = (1, 2), padding='same')(x)
	# x = convolution(x, 256, 3)
	# x = convolution(x, 256, 3)
	# x = convolution(x, 512, 3)
	# x = MaxPooling2D((3, 3), strides = 2, padding='same')(x)
	# x = Flatten()(x)
	# x = Dense(512, activation='relu')(x)
	# encoded_audio = Dense(256, activation='relu')(x)


	# # Identity encoder
	# dim = num_still_images*3 + 3
	# input_identity = Input(shape=(112, 112, dim)) 
	
	# x = convolution(input_identity, 96, 7, 2)
	# x_skip1 = MaxPooling2D((3, 3), strides=2, padding='same')(x)
	# x_skip2 = convolution(x_skip1, 256, 5, 2)
	# x_skip3 = MaxPooling2D((3, 3), strides=2, padding='same')(x_skip2)
	# x = convolution(x_skip3, 512, 3)
	# x = convolution(x, 512, 3)
	# x = convolution(x, 512, 3)
	# x = Flatten()(x)
	# x = Dense(512, activation='relu')(x)
	# encoded_identity = Dense(256, activation='relu')(x)

	# # Concatenate the audio and identity features
	# concatenated_features = concatenate([encoded_audio, encoded_identity])

	# # Decoder
	# x = Dense(98, activation='relu')(concatenated_features)
	# x = Reshape((7, 7, 2))(x)
	# x = transposed_convolution(x, 512, 6)
	# x = transposed_convolution(x, 256, 5)
	# x = concatenate([x, x_skip3])
	# x = transposed_convolution(x, 96, 5, 2)
	# x = concatenate([x, x_skip2])
	# x = transposed_convolution(x, 96, 5, 2)
	# x = concatenate([x, x_skip1])
	# x = transposed_convolution(x, 64, 5, 2)
	# decoded = Conv2DTranspose(3, (5, 5), strides=2, activation='sigmoid', padding='same')(x)

	# model = Model(inputs = [input_audio, input_identity], outputs = [decoded])
 
	# try:
		# model = ModelMGPU(model, gpus)
	# except:
		# pass

	# model.compile(optimizer='adam', loss='mean_absolute_error')
	# print(model.summary())

	# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

	# if not os.path.exists(model_dir):
	# 	os.makedirs(model_dir)


	# early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
	# checkpoint = ModelCheckpoint(model_dir+'/best_model.h5', monitor='val_loss', mode='min', \
	# 							verbose=1, save_best_only=True)


	# history = model.fit_generator(generator=training_generator,
	# 							validation_data=validation_generator,
	# 							steps_per_epoch=steps_per_epoch,
	# 							epochs=epochs,
	# 							use_multiprocessing=False,
	# 							callbacks=[tensorboard_callback, early_stopping, checkpoint]
	# 							)



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

	