from keras.models import load_model
import argparse
import pickle
import numpy as np
from scripts.utils import *
import subprocess
from numpy import newaxis
from scripts.face_utils import *
from scripts.faces import *
from scripts.mfcc_features import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import __add__

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

def extract_test_features(test_file, test_img, feature_dir, fps=25):

	print("Test file: ", test_file)
	# time = 0.0
	# time = subprocess.Popen("ffprobe -hide_banner -loglevel panic -i %s -show_format -v quiet \
	# 						| sed -n 's/duration=//p'" % (test_file) , stdout=subprocess.PIPE, \
	# 						shell=True)
	
	cap = cv2.VideoCapture(f"{test_file}")
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()

	# time = float(time.stdout.read())
	# print("Duration: ", time)
	# num_frames = int(fps * time)

	test_dir = feature_dir + "/" + test_file.split("/")[-1].split(".")[0]
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)


	# Extract the image features
	img = read_image(test_img)
	img_feature = detect_face(img)
	img_masked = img_feature.copy()
	lower_index = img_masked.shape[0]//2
	upper_index = img_masked.shape[0]
	img_masked[lower_index:upper_index, :upper_index] = [0,0,0]	
	concatenated_feature = np.dstack((img_feature, img_masked))

	#concatenated_feature = np.dstack((img_feature, img_feature, img_feature, img_feature, \
	#									img_feature, img_masked))

	

	# Extract theaudio features
	mfcc = extract_mfcc(test_file)
	
	fname  = test_dir + "/" + test_file.split("/")[-1].split(".")[0] + '.pkl'
	with open(fname, "wb") as file:
		pickle.dump(mfcc, file)

	audio_features = []
	identity_features = []
	
	for i in range(num_frames):
		time = (i+1) / fps
		start_time = time - 0.175
		if(start_time < 0):
			continue
		start_mfcc = int(start_time*100)
		feature = mfcc[:, start_mfcc:(start_mfcc + 35)]
		if(feature.shape != (12, 35)):
			continue

		audio_features.append(feature)
		identity_features.append(concatenated_feature)	    
		if((i+35) > mfcc.shape[1]):
			break
		
	x_test_audio = np.array(audio_features)
	x_test_audio = x_test_audio.astype('float32') 
	x_test_audio = x_test_audio[:, :, :, newaxis]
	print("Audio shape: ", x_test_audio.shape)

	x_test_identity = np.array(identity_features)
	x_test_identity = x_test_identity.astype('float32') / 255.
	print("Identity shape: ", x_test_identity.shape)

	return  torch.as_tensor(x_test_audio).permute(0, 3, 1, 2), \
			torch.as_tensor(x_test_identity).permute(0, 3, 1, 2)


def predict(model, test_file, x_test_audio, x_test_identity, feature_dir):
	predicted_frames = model(x_test_audio, x_test_identity)
	print("Predicted frames: ", predicted_frames.shape)

	test_dir = feature_dir + "/" + test_file.split("/")[-1].split(".")[0] + '/frames' 
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	predictions = []
	for i in range(len(predicted_frames)):
		im = np.clip(np.round((predicted_frames[i].permute(1, 2, 0).detach().numpy()*255).astype(np.uint8)), 0, 255)
		predictions.append(im)
		fname = test_dir + "/" + test_file.split("/")[-1].split(".")[0] + '_' + str(i+1)
		write_image(im, fname)

	return predictions


def generate_video(test_file, predictions, output_file_name, fps=25):

	height, width, layers = predictions[0].shape
	fname = 'output.avi' 
	video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
 
	for i in range(len(predictions)):
		img = cv2.cvtColor(predictions[i], cv2.COLOR_BGR2RGB)
		video.write(np.uint8(img))
	
	video.release()

	video_output = output_file_name + '.mkv'
	subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -i %s -c copy -map 0:v -map 1:a %s' % \
													(fname, test_file, video_output), shell=True) 


	os.remove("output.avi")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-m', '--model_path', help='Saved Model path', default="saved_models/best_model_pytorch.pth")	
	parser.add_argument('-t', '--test_file', help='Test audio or video file', default="1.mp4")
	parser.add_argument('-f', '--test_frame', help='Test frame', default="frame.jpg")
	parser.add_argument('-fd', '--test_feature_dir', default='test_features', required=False, \
						help='Path to dave test features')
	parser.add_argument('-o', '--output_file_name', default='output_video', required=False, \
						help='Name of the output video file')
	args = parser.parse_args()

	model = TalkingHeadModel(1)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()


	x_test_audio, x_test_identity = extract_test_features(args.test_file, args.test_frame, \
															args.test_feature_dir)

	predictions = predict(model, args.test_file, x_test_audio, x_test_identity, args.test_feature_dir)

	generate_video(args.test_file, predictions, args.output_file_name)
