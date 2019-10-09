import numpy as np
import cv2


class Image(object):
    def __init__(self, id, background_path, texture_path):
        self.id = id

        self.background = cv2.imread(background_path)
        self.texture = cv2.imread(texture_path)
        self.randomize_flips()

        self.shape = self.background.shape

    def randomize_flips(self):
        background_flip_code = np.random.randint(-1,3)
        texture_flip_code = np.random.randint(-1,3)
        if background_flip_code != 2:
            self.background = cv2.flip(self.background, background_flip_code)
        if texture_flip_code != 2:
            self.texture = cv2.flip(self.texture, texture_flip_code)

    def get_random_location(self):
        x = np.random.randint(self.shape[1])
        y = np.random.randint(self.shape[0])
        return x, y

    def get_random_points(self, sigma, num_points, x, y):
        ptsx, ptsy = (list() for x in range(2))
        for _ in range(num_points):
            ptsx.append(int(np.random.normal(x, sigma)))
            ptsy.append(int(np.random.normal(y, sigma)))
        return ptsx, ptsy

    def draw_dots(self, ptsx, ptsy, radius, mask):
        for ptx, pty in zip(ptsx, ptsy):
            if ptx >= radius and pty >= radius and ptx < self.shape[1] - radius and pty < self.shape[0] - radius:
                cv2.circle(mask, (ptx, pty), radius, (255,255,255), -1)
        return mask

    def morph_dots(self, kernel, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        return mask

class SemanticImage(Image):
    def __init__(self, id, background_path, texture_path):
        super().__init__(id, background_path, texture_path)

        self.image = None
        self.mask = np.zeros(self.shape[:2])

    def create_mask(self, num_locations, sigma, num_points, radius, crop_dim):
        for _ in range(num_locations):
            x, y = self.get_random_location()
            ptsx, ptsy = self.get_random_points(sigma, num_points, x, y)
            self.mask = self.draw_dots(ptsx, ptsy, radius, self.mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30))
        self.mask = self.morph_dots(kernel, self.mask)

        if crop_dim:
            pass


#class InstanceImage(Image):

if __name__ == '__main__':
    import os
    import argparse

    ROOT_DIR = os.path.abspath('../Generators')

    parser = argparse.ArgumentParser(
        description="Generate random rust data from rust texture")
    parser.add_argument('command', type=str,
                        help="'instance' or 'semantic'")
    parser.add_argument('--backgrounds_path', default=os.path.join(ROOT_DIR, 'backgrounds'), type=str,
                        help="path to background directory")
    parser.add_argument('--textures_path', default=os.path.join(ROOT_DIR, 'rust_textures'), type=str,
                        help="path to texture directory")
    parser.add_argument('--num_images', default=None, type=int,
                        help="number of images to be generated")
    args = parser.parse_args()

    ##Creating images
    background_file = np.random.choice(os.listdir(args.backgrounds_path))
    texture_file = np.random.choice(os.listdir(args.textures_path))

    background_path = os.path.join(args.backgrounds_path, background_file)
    texture_path = os.path.join(args.textures_path, texture_file)

    image = SemanticImage(1, background_path, texture_path)
    image.create_mask(5, 20, 500, 1, 512)

    cv2.imshow('texture', image.mask)
    cv2.waitKey(0)
