import numpy as np
from .utils import *
import random
from numpy import newaxis
import pickle

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class TalkingFaceDataset(Dataset):
    def __init__(self, keys, num_still_images, batch_size=64, fps=25, shuffle=True):
        self.keys = keys
        self.num_still_images = num_still_images
        self.batch_size = batch_size
        self.fps = fps

        self.shuffle = shuffle
        self.on_epoch_end()

        self.transform = transforms.ToTensor()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if len(indexes) == 0:
            return [], [], []

        # # Find list of IDs
        batch_keys = [self.keys[k] for k in indexes]
        fname = self.keys[index]
        # # Generate data
        X_a, X_i, y = self.__data_generation(batch_keys)

        return X_a, X_i, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.keys))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_keys):
        'Generates data containing batch_size samples' 
        
        X_audio = []
        X_identity = []
        Y_train = []

        # Generate data
        for fname in batch_keys:
            frame_num = get_frame_num(fname)
            
            time = frame_num / self.fps
            start_time = time - 0.175
            if(start_time < 0):
                continue

            img_gt = read_image(fname)
            # Mask the lower part of the ground truth image
            img_gt_masked = img_gt.copy()
            lower_index = img_gt_masked.shape[0]//2
            upper_index = img_gt_masked.shape[0]
            img_gt_masked[lower_index:upper_index, :upper_index] = [0,0,0] 

            dir_name = os.path.dirname(fname)
            all_frames = get_files(dir_name, extension=['.jpg'])
            
            if(self.num_still_images == 1):
                selected_frames = [frame for frame in all_frames if \
                                        np.abs(frame_num - get_frame_num(frame)) > 5]
                if(len(selected_frames) < 1):
                    continue

                feature_fname = random.choice(selected_frames)
                img_feature = read_image(feature_fname)
                            
                # Concatenate the feature and the ground truth images channel wise
                identity_feature = np.dstack((img_feature, img_gt_masked))
                if(identity_feature.shape != (112, 112, 6)):
                    continue

            else:
                if(len(all_frames) < self.num_still_images):
                    continue

                selected_frames = [frame for frame in all_frames if \
                                        np.abs(frame_num - get_frame_num(frame)) > 5]

                if(len(selected_frames) < self.num_still_images):
                    continue

                feature_fnames = random.sample(selected_frames, self.num_still_images)
                
                img_features = []
                for imgs in feature_fnames:
                    img_features.append(read_image(imgs))

                identity_feature = np.dstack((img_features[0], img_features[1], img_features[2], \
                                    img_features[3], img_features[4], img_gt_masked))
                if(identity_feature.shape != (112, 112, 18)):
                    continue

            start_mfcc = int(start_time*100)
            fname_audio = fname.split(".")[0].rsplit('_',1)[0] + '.pkl' 
            with open(fname_audio, "rb") as fp:   
                mfcc = pickle.load(fp) 
            #mfcc = loadmat(fname_audio)['mfccs']
            #print(mfcc.shape)

            audio_feature = mfcc[:, start_mfcc:(start_mfcc + 35)]
            if(audio_feature.shape != (12, 35)):
                continue


            x_audio = np.array(audio_feature)
            x_audio = x_audio.astype('float32') 
            x_audio = x_audio[:, :, newaxis]

            x_identity = np.array(identity_feature)
            x_identity = x_identity.astype('float32') / 255.

            y_train = np.array(img_gt)
            y_train = y_train.astype('float32') / 255.

            X_audio.append(x_audio)
            X_identity.append(x_identity)

            # Store ground truth data
            Y_train.append(y_train)

        return torch.as_tensor(np.array(X_audio)).permute(0, 3, 1, 2), \
                torch.as_tensor(np.array(X_identity)).permute(0, 3, 1, 2), \
                torch.as_tensor(np.array(Y_train)).permute(0, 3, 1, 2)