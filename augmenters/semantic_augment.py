# Semantic image segmentation
# Takes dataset on format:

# dataset
# ├── images
# └── masks
if __name__=='__main__':
    import os
    import argparse
    import imageio, cv2
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa

    parser = argparse.ArgumentParser(description=
    "Module for augmenting a folder of images")
    parser.add_argument('--dataset_path', type=str,
                        help="Path to dataset")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to save directory")
    parser.add_argument('--keep_names', type=bool, default=False,
                        help="Flag to keep names or give names based on dataset name")
    parser.add_argument('--crop', type=float, default=0,
                        help="Max percentage to crop")
    parser.add_argument('--flip_h', type=float, default=0,
                        help="Probability for randomly flipping horizontally")
    parser.add_argument('--flip_v', type=float, default=0,
                        help="Probability for randomly flipping horizontally")
    parser.add_argument('--brightness', type=float, default=0,
                        help="Max brightness change")
    parser.add_argument('--rotation', type=int, default=0,
                        help="Max rotation change")
    parser.add_argument('--scale', type=float, default=0,
                        help="Max scale change")
    parser.add_argument('--shear', type=float, default=0,
                        help="Max shear change")
    parser.add_argument('--translate_percent', type=float, default=0,
                        help="Max translation percent change")
    parser.add_argument('--num_augments', type=int, default=1,
                        help="Number of augmentations done on each image")
    args = parser.parse_args()

    if not args.save_path:
        args.save_path = os.path.split(args.dataset_path)[-1] + '_augmented'

    if not os.path.exists(os.path.join(args.save_path, 'images')):
        os.makedirs(os.path.join(args.save_path, 'images'))
        os.makedirs(os.path.join(args.save_path, 'masks'))

    seq = iaa.Sequential([
        iaa.Fliplr(args.flip_h),
        iaa.Flipud(args.flip_v),
        iaa.Crop(percent=(0, args.crop)),
        iaa.Add((-args.brightness * 100, args.brightness * 100), per_channel=True),
        iaa.Affine(
            scale={"x": (1 - args.scale, 1 + args.scale), "y": (1 - args.scale, 1 + args.scale)},
            translate_percent={"x": (-args.translate_percent, args.translate_percent), "y": (-args.translate_percent, args.translate_percent)},
            rotate=(-args.rotation, args.rotation),
            shear=(-args.shear, args.shear)
        )
    ])

    image_path = os.path.join(args.dataset_path, 'images')
    masks_path = os.path.join(args.dataset_path, 'masks')

    for i, file in enumerate(os.listdir(image_path)):
        print(f"Augmenting {file}")
        image = cv2.imread(os.path.join(image_path, file))
        mask = cv2.imread(os.path.join(masks_path, file))

        for j in range(args.num_augments):
            n = i*args.num_augments + j
            ia.seed(n)
            seq_det = seq.to_deterministic()

            aug_image = seq_det(images=[image])
            aug_mask = seq_det(images=[mask])

            if args.keep_names:
                file_name = f"{file.split['.'][0]}_{n}.png"
            else:
                file_name = f"{os.path.split(args.dataset_path)[-1]}_{n}.png"

            cv2.imwrite(os.path.join(args.save_path, 'images', file_name), aug_image[0])
            cv2.imwrite(os.path.join(args.save_path, 'masks', file_name), aug_mask[0])
