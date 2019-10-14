import os
import shutil

def path_to_path(path):
    tail = os.path.split(path)[-1]
    return tail

if __name__=='__main__':
    import argparse
    import numpy as np
    import cv2

    parser = argparse.ArgumentParser(description="Creates blank masks for a folder of images to indicate no rust.")
    parser.add_argument('--dataset_path', default=None,
                        help="Path to folder of images to be masked")
    parser.add_argument('--save_path', default=None,
                        help="Path to save folder")
    parser.add_argument('--keep_names', default=False,
                        help="Flag to keep image names")
    parser.add_argument('--save_type', default='.png',
                        help="Save type, '.png' or '.jpg'")                       
    args = parser.parse_args()

    assert args.dataset_path, "Path to data folder must be specified"

    if not args.save_path:
        args.save_path = f"{path_to_path(args.dataset_path)}_masked"
    if not os.path.exists(args.save_path):
        os.makedirs(os.path.join(args.save_path, 'images'))
        os.makedirs(os.path.join(args.save_path, 'masks'))

        print('Created directories')

    for i, file in enumerate(os.listdir(args.dataset_path)):
        image = cv2.imread(os.path.join(args.dataset_path, file))
        mask = np.zeros(image.shape[:2])
        if args.keep_names:
            name = f"{file.split('.')[0]}{args.save_type}"
        else:
            name = f"{path_to_path(args.dataset_path)}_{i}{args.save_type}"

        cv2.imwrite(os.path.join(args.save_path, 'images', name), image)
        cv2.imwrite(os.path.join(args.save_path, 'masks', name), mask)
    print(f"Finished! Saved images and masks in {args.save_path}")


