import numpy as np
import random

from basenet.vision import transforms as btransforms
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps


class Policy(object):
    def __init__(self, params, fillcolor=(128, 128, 128), image_size=32):
        """
        Get parameters from tuner to initialize the policy use for this trial
        :param params: Dictionary like parameters. Expected format:
        {
            "operation1_1": "shearX",
            "operation1_2": "rotate",
            "prob1_1": 0.4,
            "prob1_2": 0.2,
            "magnitude1_1": 0.7,
            "magnitude1_2": 0.1,
            .....
        }
        :type params: dict
        :param fillcolor: color used to fill empty space in operations like rotate
        :type fillcolor: tuple
        """
        self.policies = []
        num_policy = 5
        self.resize = transforms.Resize(image_size)

        for i in range(1, num_policy + 1):
            self.policies.append(
                SubPolicy(
                    p1=params[f'prob{i}_1'],
                    operation1=params[f'operation{i}_1'],
                    magnitude1=params[f'magnitude{i}_1'],
                    p2=params[f'prob{i}_2'],
                    operation2=params[f'operation{i}_2'],
                    magnitude2=params[f'magnitude{i}_2'],
                    fillcolor=fillcolor
                )
            )

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        img = self.policies[policy_idx](img)
        return self.resize(img)


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude1, p2, operation2, magnitude2,
                 fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": (0, 0.3),
            "shearY": (0, 0.3),
            "translateX": (0, 150 / 331),
            "translateY": (0, 150 / 331),
            "rotate": (0, 30),
            "color": (0.0, 0.9),
            "posterize": (8, 4),
            "solarize": (256, 0),
            "contrast": (0.0, 0.9),
            "sharpness": (0.0, 0.9),
            "brightness": (0.0, 0.9),
            "autocontrast": (0.0, 0.0),
            "equalize": (0.0, 0.0),
            "invert": (0.0, 0.0),
            "randomCrop": (28.0, 32.0),
            "randomHorizontalFlip": (0.0, 1.0)
        }

        self.p1 = p1
        self.p2 = p2
        self.magnitude1 = (magnitude1) * (ranges[operation1][1] - ranges[operation1][0]) + \
            ranges[operation1][0]
        if operation1 in ['posterize', 'randomCrop']:
            self.magnitude1 = np.round(self.magnitude1, 0).astype(np.int)
        self.magnitude2 = (magnitude2) * (ranges[operation2][1] - ranges[operation2][0]) + \
            ranges[operation2][0]
        if operation2 in ['posterize', 'randomCrop']:
            self.magnitude2 = np.round(self.magnitude2, 0).astype(np.int)

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when
        # -rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot)\
                .convert(img.mode)

        self.reflection_padding = btransforms.ReflectionPadding(margin=(4, 4))

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        def get_operation_func(operation, magnitude):
            if operation == 'randomCrop':
                random_crop = transforms.RandomCrop(magnitude)
                return lambda img, _: random_crop(self.reflection_padding(img))
            elif operation == 'randomHorizontalFlip':
                random_horizontal_flip = transforms.RandomHorizontalFlip(magnitude)
                return lambda img, _: random_horizontal_flip(img)
            return func[operation]

        self.operation1 = get_operation_func(operation1, self.magnitude1)
        self.operation2 = get_operation_func(operation2, self.magnitude2)

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img
