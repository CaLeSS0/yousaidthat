import shutil
import os
import sys

path = "D:/attentiongetter/dataset/voxceleb2/vids/dev/mp4/"
all_dirs = os.listdir(path)

vid_count = 0
max_vids = 1000
for i in range(len(all_dirs)):
    all_vid_dirs = os.listdir(path + all_dirs[i])
    for j in range(len(all_vid_dirs)):
        all_videos = os.listdir(path + all_dirs[i] + "/" + all_vid_dirs[j])
        
        for vid in all_videos:
            if vid_count < max_vids:
                shutil.copyfile(f"{path}/{all_dirs[i]}/{all_vid_dirs[j]}/{vid}", f"D:/attentiongetter/research/you_said_that/data/vox/{str(vid_count).zfill(6)}.mp4")
                vid_count += 1
            else:
                sys.exit()
            
            print(f"{vid_count}/{max_vids}")