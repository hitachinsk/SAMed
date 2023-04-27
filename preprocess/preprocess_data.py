import os
import time
import argparse
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
from icecream import ic


parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default='../data/Abdomen/RawData', help='download path for Synapse data')
parser.add_argument('--dst_path', type=str,
                   default='../data/Synapse', help='root dir for data')
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')
args = parser.parse_args()

test_data = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]

hashmap = {1:1, 2:2, 3:3, 4:4, 5:0, 6:5, 7:6, 8:7, 9:0, 10:0, 11:8, 12:0, 13:0}

# 1: spleen
# 2: right kidney
# 3: left kidney
# 4: gallbladder
# 5: liver
# 6: stomach
# 7: aorta
# 8: pancreas

def preprocess_train_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/train_npz", exist_ok=True)

    a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][3:7]

        if int(number) in test_data:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))  # [D, W, H]
        label_data = np.transpose(label_data, (2, 1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        for dep in range(D):
            save_path = f"{args.dst_path}/train_npz/case{number}_slice{dep:03d}.npz"
            np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])
    pbar.close()


def preprocess_valid_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/test_vol_h5", exist_ok=True)

    a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][3:7]

        if int(number) not in test_data:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            assert counter == k
            counter += 1
            label_data[label_data == k] = hashmap[k]

        save_path = f"{args.dst_path}/test_vol_h5/case{number}.npy.h5"
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()
    pbar.close()


if __name__ == "__main__":
    data_root = f"{args.src_path}/Training"

    # String sort
    image_files = sorted(glob(f"{data_root}/img/*.nii.gz"))
    label_files = sorted(glob(f"{data_root}/label/*.nii.gz"))

    # preprocess_train_image(image_files, label_files)
    preprocess_valid_image(image_files, label_files)