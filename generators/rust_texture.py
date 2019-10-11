import numpy as np
import cv2


class Image(object):
    def __init__(self, id, background_path, texture_path):
        self.id = id

        self.texture = cv2.imread(texture_path)
        self.background = cv2.imread(background_path)
        self.background = cv2.resize(self.background, (self.texture.shape[1], self.texture.shape[0]))
        self.randomize_flips()

        self.shape = self.texture.shape

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
        self.mask = np.zeros(self.shape[:2], dtype='uint8')

    def create_mask(self, num_locations, sigma, num_points, radius):
        for _ in range(num_locations):
            x, y = self.get_random_location()
            ptsx, ptsy = self.get_random_points(sigma, num_points, x, y)
            self.mask = self.draw_dots(ptsx, ptsy, radius, self.mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 5))
        self.mask = self.morph_dots(kernel, self.mask)

    def random_crop(self, crop_dim):
        startx = np.random.randint(0, self.shape[1] - crop_dim)
        starty = np.random.randint(0, self.shape[0] - crop_dim)
        self.mask = self.mask[starty:starty + crop_dim, startx:startx + crop_dim]
        self.texture = self.texture[starty:starty + crop_dim, startx:startx + crop_dim]
        self.background = self.background[starty:starty + crop_dim, startx:startx + crop_dim]

    def create_image(self, num_locations, sigma, num_points, radius, crop_dim):
        self.create_mask(num_locations, sigma, num_points, radius)
        if crop_dim:
            self.random_crop(crop_dim)
            
        mask_inv = cv2.bitwise_not(self.mask)

        fg = cv2.bitwise_and(self.texture, self.texture, mask=self.mask)
        bg = cv2.bitwise_and(self.background, self.background, mask=mask_inv)

        self.image = cv2.bitwise_or(fg, bg)


#class InstanceImage(Image):

if __name__ == '__main__':
    import os
    import argparse

    NUM_LOCATIONS = 5
    NUM_POINTS = 2000
    SIGMA = 60
    RADIUS = 1

    parser = argparse.ArgumentParser(
        description="Generate random rust data from rust texture")
    parser.add_argument('command', type=str,
                        help="'instance' or 'semantic'")
    parser.add_argument('--backgrounds_path', default='backgrounds', type=str,
                        help="Path to background directory")
    parser.add_argument('--textures_path', default='rust_textures', type=str,
                        help="Path to texture directory")
    parser.add_argument('--save_path', default='generated', type=str,
                        help="Path to save directory")   
    parser.add_argument('--num_images', default=1, type=int,
                        help="Number of images to be generated") 
    parser.add_argument('--start_iter', default=0, type=int,
                        help="Start iteration for generation. Useful for generating over multiple sessions")
    parser.add_argument('--out_dim', default=512, type=int,
                        help="Dimension of output. If set to 0, the original size of texture will be preserved")         
    parser.add_argument('--save_type', default='.png', type=str,
                        help="Save type .png/.jpg")
    args = parser.parse_args()

    # Setting up directories
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'images')):
        os.mkdir(os.path.join(args.save_path, 'images'))
    if not os.path.exists(os.path.join(args.save_path, 'masks')):
        os.mkdir(os.path.join(args.save_path, 'masks'))

    # Creating images
    for i in range(args.start_iter, args.start_iter + args.num_images):
        background_file = np.random.choice(os.listdir(args.backgrounds_path))
        texture_file = np.random.choice(os.listdir(args.textures_path))

        background_path = os.path.join(args.backgrounds_path, background_file)
        texture_path = os.path.join(args.textures_path, texture_file)

        image = SemanticImage(args, background_path, texture_path)
        image.create_image(NUM_LOCATIONS, SIGMA, NUM_POINTS, RADIUS, args.out_dim)

        cv2.imwrite("{}/images/rust{}{}".format(args.save_path, i, args.save_type), image.image)
        cv2.imwrite("{}/masks/rust{}{}".format(args.save_path, i, args.save_type), image.mask)
        
        print('Saved image {}'.format(i))


