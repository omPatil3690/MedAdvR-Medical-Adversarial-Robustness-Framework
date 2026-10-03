import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from .augmentation import SegmentationAugmentation

class TumorDataset(Dataset):
    def __init__(self, root_dir, img_size , augment =None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment


        self.images = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".png") and not f.endswith("_mask.png")
        ])

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = img_name.replace(".png", "")
        mask_name = f"{base}_mask.png"

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.augment is not None:
            image, mask = self.augment(image, mask)

        return image, mask


class ExtractedDataset(Dataset):
    def __init__(self, root_dir, img_size, augment=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment

        # Assuming standard structure where images are in 'images' folder and masks in 'masks' folder
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        
        if os.path.exists(self.images_dir) and os.path.exists(self.masks_dir):
            self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            # Fallback if they are in root_dir directly but have corresponding masks somewhere, user might need to adjust
            self.images_dir = root_dir
            self.masks_dir = root_dir
            self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and 'mask' not in f.lower()])

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # If mask has exactly same name or with '_mask' suffix
        base_name, ext = os.path.splitext(img_name)
        mask_name = img_name
        if not os.path.exists(os.path.join(self.masks_dir, mask_name)):
            if os.path.exists(os.path.join(self.masks_dir, f"{base_name}_mask{ext}")):
                mask_name = f"{base_name}_mask{ext}"
            elif os.path.exists(os.path.join(self.masks_dir, f"{base_name}_mask.png")):
                mask_name = f"{base_name}_mask.png"
            elif os.path.exists(os.path.join(self.masks_dir, f"{base_name}.png")):
                mask_name = f"{base_name}.png"

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        try:
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = np.array(mask)
            mask = (mask > 128).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
        except Exception:
            # Fallback if mask not found
            mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        if self.augment is not None:
            image, mask = self.augment(image, mask)

        return image, mask


class DicomDataset(Dataset):
    def __init__(self, root_dir, img_size, augment=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment

        self.images_dir = os.path.join(root_dir, "images") if os.path.exists(os.path.join(root_dir, "images")) else root_dir
        self.masks_dir = os.path.join(root_dir, "masks") if os.path.exists(os.path.join(root_dir, "masks")) else root_dir

        self.images = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(('.dcm', '.dicom'))])

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import pydicom
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        try:
            ds = pydicom.dcmread(img_path)
            img_arr = ds.pixel_array
            # Normalize to 0-255 safely
            img_min, img_max = np.min(img_arr), np.max(img_arr)
            if img_max > img_min:
                img_arr = (img_arr - img_min) / (img_max - img_min)
            img_arr = (img_arr * 255).astype(np.uint8)

            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr]*3, axis=-1)

            image = Image.fromarray(img_arr).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.img_size, self.img_size))

        image = self.img_transform(image)

        # Mask parsing
        base_name, _ = os.path.splitext(img_name)
        mask_opts = [
             f"{base_name}_mask.png", f"{base_name}.png", f"{base_name}_mask.dcm", f"{base_name}.dcm"
        ]

        mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        for cand in mask_opts:
            cand_path = os.path.join(self.masks_dir, cand)
            if os.path.exists(cand_path):
                if cand_path.endswith('.dcm'):
                     ds_m = pydicom.dcmread(cand_path)
                     m_arr = ds_m.pixel_array
                     m_arr = (m_arr > 0).astype(np.float32)
                     m_img = Image.fromarray(m_arr)
                     m_img = m_img.resize((self.img_size, self.img_size), Image.NEAREST)
                     mask = torch.from_numpy(np.array(m_img)).unsqueeze(0)
                else:
                     m_img = Image.open(cand_path).convert('L')
                     m_img = m_img.resize((self.img_size, self.img_size), Image.NEAREST)
                     m_arr = np.array(m_img)
                     m_arr = (m_arr > 128).astype(np.float32)
                     mask = torch.from_numpy(m_arr).unsqueeze(0)
                break

        if self.augment is not None:
             image, mask = self.augment(image, mask)

        return image, mask


class NiftiDataset(Dataset):
    def __init__(self, root_dir, img_size, augment=None):
        import nibabel as nib
        self.img_size = img_size
        self.augment = augment

        img_dir_candidates = ["imagesTr", "images", ""]
        mask_dir_candidates = ["labelsTr", "labels", "masks", ""]

        self.images_dir = None
        for d in img_dir_candidates:
            cand = os.path.join(root_dir, d) if d else root_dir
            if os.path.exists(cand) and any(f.endswith('.nii.gz') for f in os.listdir(cand)):
                self.images_dir = cand
                break

        self.masks_dir = None
        for d in mask_dir_candidates:
            cand = os.path.join(root_dir, d) if d else root_dir
            if os.path.exists(cand) and any(f.endswith('.nii.gz') for f in os.listdir(cand)):
                self.masks_dir = cand
                break

        if not self.images_dir:
            raise ValueError("No NiFTI images found in the dataset folder.")

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.nii.gz') and not f.startswith('._')])

        self.slice_index_map = []
        for f in self.image_files:
            img_path = os.path.join(self.images_dir, f)
            nii = nib.load(img_path)
            shape = nii.shape
            depth = shape[2] if len(shape) >= 3 else 1
            for d in range(depth):
                self.slice_index_map.append((f, d))

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.slice_index_map)

    def _get_mask_filename(self, img_name):
        if not self.masks_dir:
            return img_name
        if os.path.exists(os.path.join(self.masks_dir, img_name)):
            return img_name
        base = img_name.replace('_0000.nii.gz', '.nii.gz')
        if os.path.exists(os.path.join(self.masks_dir, base)):
            return base
        return img_name

    def __getitem__(self, idx):
        import nibabel as nib
        file_name, slice_idx = self.slice_index_map[idx]

        img_path = os.path.join(self.images_dir, file_name)
        
        nii_img = nib.load(img_path)
        img_vol = nii_img.dataobj[..., slice_idx] 
        img_vol = np.array(img_vol)

        img_min, img_max = np.min(img_vol), np.max(img_vol)
        if img_max > img_min:
             img_vol = (img_vol - img_min) / (img_max - img_min)
        img_vol = (img_vol * 255).astype(np.uint8)

        if img_vol.ndim == 3 and img_vol.shape[2] == 4:
            img_vol = img_vol[:, :, :3]
        elif img_vol.ndim == 2:
            img_vol = np.stack([img_vol]*3, axis=-1)

        image = Image.fromarray(img_vol).convert('RGB')
        image = self.img_transform(image)

        mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        if self.masks_dir:
             mask_name = self._get_mask_filename(file_name)
             mask_path = os.path.join(self.masks_dir, mask_name)
             try:
                 nii_mask = nib.load(mask_path)
                 mask_vol = nii_mask.dataobj[..., slice_idx]
                 mask_vol = np.array(mask_vol)
                 mask_vol = (mask_vol > 0).astype(np.float32)
                 m_img = Image.fromarray(mask_vol)
                 m_img = m_img.resize((self.img_size, self.img_size), Image.NEAREST)
                 mask = torch.from_numpy(np.array(m_img)).unsqueeze(0)
             except Exception:
                 pass

        if self.augment is not None:
             image, mask = self.augment(image, mask)

        return image, mask


class DatasetLoader:
    def __init__(self, root_dir, dataset_type="tumor", img_size=128, batch_size=16, shuffle=True, num_workers=4, pin_memory=True , augment = False):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.augment = SegmentationAugmentation(img_size) if augment else None

        if self.dataset_type.lower() == "extracted":
            self.dataset = ExtractedDataset(
                self.root_dir,
                self.img_size,
                augment=self.augment
            )
        elif self.dataset_type.lower() == "dicom":
            self.dataset = DicomDataset(
                self.root_dir,
                self.img_size,
                augment=self.augment
            )
        elif self.dataset_type.lower() == "nifti":
            self.dataset = NiftiDataset(
                self.root_dir,
                self.img_size,
                augment=self.augment
            )
        else:
            self.dataset = TumorDataset(
                self.root_dir, 
                self.img_size,
                augment=self.augment
            )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.loader)

    def get_loader(self):
        return self.loader
