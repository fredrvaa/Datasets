import cv2
import os

def path_to_name(path):
    tail = os.path.split(path)[-1]
    filename = tail.rsplit( ".", 1 )[0]
    return filename

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a video into multiple images")
    parser.add_argument('--video_path', type=str, default=None,
                        help="Path to video you want to convert")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to folder you want the images to be saved to")
    parser.add_argument('--save_type', type=str, default='.png',
                        help="'.jpg' or '.png'")
    args = parser.parse_args()

    assert args.video_path, "Video path must be specified"

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = path_to_name(args.video_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    cap = cv2.VideoCapture(args.video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
           cv2.imwrite('{}/{}_{}{}'.format(save_path, save_path, i, args.save_type), frame)
        else:
            break
        print('Saved image {}'.format(i))
        i += 1

    cap.release()
    cv2.destroyAllWindows()
