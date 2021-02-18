from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

class Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        operations: 5x2 list
        probs: 5x2 list
        magnitudes: 5x2 list
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, operations, probs, magnitudes, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(probs[0][0], operations[0][0], magnitudes[0][0], 
                probs[0][1], operations[0][1], magnitudes[0][1], fillcolor),
            SubPolicy(probs[1][0], operations[1][0], magnitudes[1][0], 
                probs[1][1], operations[1][1], magnitudes[1][1], fillcolor),
            SubPolicy(probs[2][0], operations[2][0], magnitudes[2][0], 
                probs[2][1], operations[2][1], magnitudes[2][1], fillcolor),
            SubPolicy(probs[3][0], operations[3][0], magnitudes[3][0], 
                probs[3][1], operations[3][1], magnitudes[3][1], fillcolor),
            SubPolicy(probs[4][0], operations[4][0], magnitudes[4][0], 
                probs[4][1], operations[4][1], magnitudes[4][1], fillcolor),
        ]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude1, p2, operation2, magnitude2, fillcolor=(128, 128, 128)):
        # ranges = {
        #     "shearX": np.linspace(0, 0.3, 10),
        #     "shearY": np.linspace(0, 0.3, 10),
        #     "translateX": np.linspace(0, 150 / 331, 10),
        #     "translateY": np.linspace(0, 150 / 331, 10),
        #     "rotate": np.linspace(0, 30, 10),
        #     "color": np.linspace(0.0, 0.9, 10),
        #     "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
        #     "solarize": np.linspace(256, 0, 10),
        #     "contrast": np.linspace(0.0, 0.9, 10),
        #     "sharpness": np.linspace(0.0, 0.9, 10),
        #     "brightness": np.linspace(0.0, 0.9, 10),
        #     "autocontrast": [0] * 10,
        #     "equalize": [0] * 10,
        #     "invert": [0] * 10
        ranges = {
            "shearX": (0, 0.3),
            "shearY": (0, 0.3),
            "translateX": (0, 150 / 331),
            "translateY": (0, 150 / 331),
            "rotate": (0, 30),
            "color": (0.0, 0.9),
            "posterize": (8, 4), #np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": (256, 0),
            "contrast": (0.0, 0.9),
            "sharpness": (0.0, 0.9),
            "brightness": (0.0, 0.9),
            "autocontrast": (0.0, 0.0),
            "equalize": (0.0, 0.0),
            "invert": (0.0, 0.0)       
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
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
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }
        
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = (magnitude1)*(ranges[operation1][1] - ranges[operation1][0]) + ranges[operation1][0]
        if operation1 == 'posterize':
            self.magnitude1 = np.round(self.magnitude1, 0).astype(np.int)
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = (magnitude2)*(ranges[operation2][1] - ranges[operation2][0]) + ranges[operation2][0]
        if operation2 == 'posterize':
            self.magnitude2 = np.round(self.magnitude2, 0).astype(np.int)


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img