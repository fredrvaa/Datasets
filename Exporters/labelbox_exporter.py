import requests
import json
import os
import numpy as np
import imageio
import requests
from PIL import Image
from io import BytesIO

def url_to_image(url):
    res = requests.get(url)
    image = np.array(Image.open(BytesIO(res.content)))
    return image

def write_instance_masks(masks_data, mask_path):
    for j, mask_data in enumerate(masks_data):
        mask_url = mask_data['instanceURI']
        mask = url_to_image(mask_url)
        imageio.imwrite("{}/{}.png".format(mask_path, j), mask)

def write_semantic_mask(shape, masks_data, mask_path):
    mask = np.zeros((shape[0], shape[1], 4), dtype='uint8')
    for mask_data in masks_data:
        mask_url = mask_data['instanceURI']
        curr_mask = url_to_image(mask_url)
        mask = np.bitwise_or(mask, curr_mask)

    imageio.imwrite(mask_path, mask)
    print("Saved mask")

def write_images(data, num_train, args):
    for i, image_data in enumerate(data[args.start:], start = args.start):
        # Selecting train or validation
        if i <= num_train:
            subset = 'train'
        else:
            subset = 'val'

        # Reading image from url
        image_url = image_data['Labeled Data']
        image = imageio.imread(image_url)

        # Writing image
        image_path = '{}/{}/images/{}.{}'.format(args.save_folder, subset, i, args.file_type)
        imageio.imwrite(image_path, image)

        # Writing masks
        masks_data = image_data['Label']['objects']    

        if args.command == 'semantic':
            mask_path = '{}/{}/masks/{}.png'.format(args.save_folder, subset, i)
            write_semantic_mask(image.shape, masks_data, mask_path)

        elif args.command == 'instance':
            # Creating mask directories
            mask_path = 'dataset/{}/masks/{}'.format(subset, i)
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)

            write_instance_masks(masks_data, mask_path)

        print("Saved image {}".format(i))


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Export labelbox data to instance or semantic segmentation masks")
    parser.add_argument('command',
                        help="'instance' or 'semantic'")
    parser.add_argument('--split', default=0.1,
                        help="Train/validation split, float in range 0 to 1.")
    parser.add_argument('--start', default=0,
                        help="Image to start exporting from. Helpful if exporting fails at some point.")
    parser.add_argument('--data',
                        help="Path Labelbox json data")
    parser.add_argument('--save_folder', default = 'dataset',
                        help="Path to save folder")
    parser.add_argument('--file_type', default='png',
                        help="File type as jpg or png")
    args = parser.parse_args()

    # Validate arguments
    if args.command == 'instance' or args.command == 'semantic':
        assert args.data, "Argument --data is required for exporting"
        assert 0<= float(args.split) <= 1, "Argument --split must be between 0 and 1"

    args.split = float(args.split)
    args.start = int(args.start)
    # Creating dataset directories
    if not os.path.exists(args.save_folder):
        os.makedirs('{}/train/images'.format(args.save_folder))
        os.makedirs('{}/val/images'.format(args.save_folder))
        os.makedirs('{}/train/masks'.format(args.save_folder))
        os.makedirs('{}/val/masks'.format(args.save_folder))
    print("Created directories")
    # Loading data
    with open(args.data) as json_file:
        data = json.load(json_file)    

    # Train/val split
    num_images = len(data)
    num_train = num_images - int(num_images * args.split)
    
    # Writing images and masks
    write_images(data, num_train, args)

    


