import argparse
import os
import numpy as np
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm


def main(args):
    root, output, ds_res = args.root, args.output, args.ds_res
    output = output + f'_{str(ds_res)}'
    if not os.path.exists(output):
        os.makedirs(output)
    ct_slices = os.listdir(root)
    p_bar = tqdm(ct_slices)
    for ct_slice in p_bar:
        slice_data = np.load(os.path.join(root, ct_slice))
        image_data, label_data = slice_data['image'], slice_data['label']
        h, w = image_data.shape
        image_data = zoom(image_data, (ds_res / h, ds_res / w), order=3)
        label_data = zoom(label_data, (ds_res / h, ds_res / w), order=0)
        out_path = os.path.join(output, ct_slice)
        np.savez(out_path, image=image_data, label=label_data)
    p_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/LarryXu/Synapse/preprocessed_data/train_npz/')
    parser.add_argument('--output', type=str, default='/data/LarryXu/Synapse/preprocessed_data/train_npz_new')
    parser.add_argument('--ds_res', type=int, default=224, help='Downsample resulution')
    args = parser.parse_args()
    main(args)
