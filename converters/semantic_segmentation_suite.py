# CONVERTS IMAGES AND MASKS WITH FOLDER STRUCTURE:
#
# dataset
# ├── images
# └── masks
#
# TO:
#
# dataset
# ├── test
# ├── test_labels
# ├── train
# ├── train_labels
# ├── val
# └── val_labels
#
# ACCORDING TO https://github.com/fredrvaa/Semantic-Segmentation-Suite

import os
import shutil
import random

def path_to_path(path):
    tail = os.path.split(path)[-1]
    return tail

def move_files(dataset_path, save_path, split): 
    image_path = os.path.join(dataset_path, 'images')
    masks_path = os.path.join(dataset_path, 'masks')

    for file in os.listdir(image_path):
        subset = random.choices(population=['test', 'train', 'val'], weights=split, k=1)[0]
        shutil.move(os.path.join(image_path, file), os.path.join(save_path, subset,file))
        shutil.move(os.path.join(masks_path, file), os.path.join(save_path, '{}_labels'.format(subset),file))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Converts images and masks for semantic segmentation according to https://github.com/fredrvaa/Semantic-Segmentation-Suite"
    )
    parser.add_argument('--dataset_path', type=str,
                        help="Path to dataset")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to save directory")
    parser.add_argument('--split', type=str, default='0.1/0.7/0.2',
                        help="Dataset split on the form 'test/train/val'. Example: '0.1/0.7/0.2'. Must add to 1")
    args = parser.parse_args()

    args.split = [float(i) for i in args.split.split("/")]

    assert sum(args.split) == 1, "Split must add to 1"

    if not args.save_path:
        args.save_path = '{}_semseg'.format(path_to_path(args.dataset_path))
    if not os.path.exists(args.save_path):
        os.makedirs(os.path.join(args.save_path, 'test'))
        os.makedirs(os.path.join(args.save_path, 'test_labels'))
        os.makedirs(os.path.join(args.save_path, 'train'))
        os.makedirs(os.path.join(args.save_path, 'train_labels'))
        os.makedirs(os.path.join(args.save_path, 'val'))
        os.makedirs(os.path.join(args.save_path, 'val_labels'))
        print('Created directories')

    move_files(args.dataset_path, args.save_path, args.split)

    

