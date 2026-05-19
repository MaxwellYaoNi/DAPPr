import numpy as np
import os, argparse, h5py
from pathlib import Path
from tqdm import tqdm

def convert_to_hdf5(source_path, image_sub_path_list, target_path):
    h5_file = h5py.File(target_path, 'w')
    for image_path in tqdm(image_sub_path_list, mininterval=2):
        with open(os.path.join(source_path, image_path), 'rb') as f:
            img_obj = np.asarray(f.read())
        h5_file.create_dataset(image_path, data=img_obj)
    h5_file.close()

def traverse_dir(dir_path, prefix=""):
    file_list = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    for sub_path in os.listdir(dir_path):
        if sub_path.startswith('.'): continue
        full_path = os.path.join(dir_path, sub_path)
        full_sub_path = prefix + sub_path
        if os.path.isdir(full_path):
            file_list.extend(traverse_dir(full_path, full_sub_path + "/"))
        elif Path(sub_path).suffix.lower() in image_exts:
            file_list.append(full_sub_path)
    return file_list

def image_folder_to_hdf5(source_path, target_path):
    all_paths = traverse_dir(source_path)
    convert_to_hdf5(source_path, all_paths, target_path)

def traverse_hdf5(h5_group, prefix=""):
    file_list = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    for key in h5_group:
        item = h5_group[key]
        path = f"{prefix}{key}"
        if isinstance(item, h5py.Group):
            file_list.extend(traverse_hdf5(item, path+"/"))
        elif Path(key).suffix.lower() in image_exts:
            file_list.append(path)
    return file_list

def read_hdf5(root):
    h5_file = h5py.File(root, 'r')
    file_list = traverse_hdf5(h5_file)
    return file_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert jpeg images to jpeg bytes stream list and store into hdf5 file')
    parser.add_argument('--src',  type=str, default=None, required=True, help='directory path of images')
    parser.add_argument('--dst',  type=str, default=None, required=True, help='hdf5 file path')

    args = parser.parse_args()
    image_folder_to_hdf5(args.src, args.dst)
    num_images = len(read_hdf5(args.dst))
    print(f"Successfully converted {args.src} to HDF5 file {args.dst}, containing {num_images} images.")
