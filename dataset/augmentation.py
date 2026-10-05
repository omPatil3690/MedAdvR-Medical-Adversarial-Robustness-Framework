import random
import torchvision.transforms.functional as TF

class SegmentationAugmentation:
    def __init__(self, img_size=128):
        self.img_size = img_size

    def __call__(self, image, mask):
        # -------------------------
        # 🔁 Geometric Transforms
        # -------------------------

        # Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical Flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Rotation (-20° to 20°)
        angle = random.uniform(-20, 20)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # -------------------------
        # 🎨 Photometric (image only)
        # -------------------------

        # Brightness
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))

        # Contrast
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        # -------------------------
        # 🔒 Ensure mask is binary
        # -------------------------
        mask = (mask > 0.5).float()

        return image, mask