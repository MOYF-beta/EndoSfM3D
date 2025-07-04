from __future__ import absolute_import, division, print_function
import os
import random
import numpy as np
from PIL import Image 
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms
DataLoaderX = torch.utils.data.DataLoader

def pil_loader(path, mode='RGB'):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if mode == 'RGB':
                return img.convert('RGB')
            elif mode == 'GRAY':
                return img.convert('I;16')
            else:
                raise ValueError("Unsupported mode: {}".format(mode))

def opencv_loader(path, mode = 'RGB'):
    if mode == 'RGB':
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    return Image.fromarray(img)

    

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders with prefetch support"""
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 min_depth=1e-2,
                 max_depth=150,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS
        self.depth_interp = Image.BILINEAR
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Color augmentation parameters
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # Initialize resize transforms
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s),
                interpolation=self.interp
            )
        self.resize_depth = transforms.Resize(
            (self.height, self.width),
            interpolation=self.depth_interp
        )
        
        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def preprocess_depth(self, inputs):
        if "depth_gt" in inputs.keys():
            depth_gt = inputs["depth_gt"]
            depth_gt = depth_gt.squeeze(0)  # Remove the channel dimension
            depth_gt = self.resize_depth(depth_gt.unsqueeze(0))  # Resize and add the channel dimension back
            depth_gt = depth_gt.squeeze(0)  # Remove the channel dimension again
            depth_gt = depth_gt.unsqueeze(0)  # Add the channel dimension back
            inputs["depth_gt"] = depth_gt

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        line = self.filenames[index].split()
        folder = line[0]
        keyframe = folder[-1]
        sequence = folder[7]

        inputs["sequence"] = torch.from_numpy(np.array(int(sequence)))
        inputs["keyframe"] = torch.from_numpy(np.array(int(keyframe)))
        
        inputs["min_depth"] = torch.from_numpy(np.array(self.min_depth))
        inputs["max_depth"] = torch.from_numpy(np.array(self.max_depth))
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
            
        inputs["frame_id"] = torch.from_numpy(np.array(frame_index))
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)


            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        inputs["depth_gt"] = torch.zeros(1, self.height, self.width)
        if self.load_depth:
            try:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
                
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"])
                self.preprocess_depth(inputs)
            except FileNotFoundError:
                # print(f'Warning: missing depth map for {folder} {frame_index} {side}')
                pass

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

class SCAREDRAWDataset(MonoDataset):
    def __init__(self, *args, load_depth=False, depth_rescale_factor=1.0, **kwargs):
        self.load_depth = load_depth
        self.depth_rescale_factor = depth_rescale_factor
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.82, 0, 0.5, 0],
                          [0, 1.02, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        
        

    def check_depth(self):
        return self.load_depth

    def get_color(self, folder, frame_index, side, do_flip):
        image_path = self.get_image_path(folder, frame_index, side)
        if not os.path.exists(image_path):
            frame_index = 0
            image_path = self.get_image_path(folder, frame_index, side)
            # print(f'Warning: missing image for {folder} {frame_index} {side}')
        color = self.loader(image_path)
        if do_flip:
            import PIL.Image as pil
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        return image_path
    
    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        depth_path = os.path.join(
            self.data_path, folder, "depth_0{}/data".format(self.side_map[side]), f_str)

        depth_gt = self.loader(depth_path, mode='GRAY')
        depth_gt = np.array(depth_gt).astype(np.float32)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt * self.depth_rescale_factor

    

if __name__ == "__main__":

    fpath = '/mnt/c/Users/14152/ZCH/Dev/datasets/C3VD_as_SCARED'
    split_file = '/mnt/c/Users/14152/ZCH/Dev/datasets/all_splits/c3vd/train_files.txt'
    ds_filenames = [line.rstrip() for line in open(split_file).readlines()]
    dataset = SCAREDRAWDataset(
        data_path=fpath, # file path
        filenames=ds_filenames,
        frame_idxs=[-1, 0, 1], # adjacent frames, default [-1, 0, 1]
        height=256,
        width=320,
        num_scales=4, # rescale factor for image pyramid
        is_train=True,
        img_ext='.png',
        load_depth=True,
        depth_rescale_factor=1
    )

    print(len(dataset))
    print(dataset[0].keys())

    import matplotlib.pyplot as plt

    # Display the color image
    color_image = dataset[0][('color', 0, 0)].permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Color Image")
    plt.imshow(color_image)

    # Display the depth map
    depth_map = dataset[0]['depth_gt'].squeeze().numpy()
    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth_map, cmap='gray')

    plt.show()