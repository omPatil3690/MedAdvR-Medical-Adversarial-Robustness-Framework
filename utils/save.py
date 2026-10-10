import os
import torch
import logging
from torchvision import transforms
from PIL import Image


def save_adv_samples(adv_samples, save_dir):
    """
    Save adversarial samples to disk.
    
    Args:
        adv_samples: List of tuples (x_adv, y_mask) where both are torch tensors
        save_dir: Directory path to save samples
    
    Returns:
        Number of samples saved
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectories
    img_dir = os.path.join(save_dir, "images")
    mask_dir = os.path.join(save_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    saved_count = 0
    
    logging.info(f"Saving {len(adv_samples)} adversarial samples to {save_dir}")
    
    for idx, (x_adv, y_mask) in enumerate(adv_samples):
        try:
            # Handle tensor shapes
            # x_adv: (C, H, W) or needs to be permuted
            if x_adv.dim() == 3 and x_adv.shape[0] in [1, 3]:
                img_tensor = x_adv
            else:
                img_tensor = x_adv
            
            # Convert to PIL Image
            img_pil = to_pil((img_tensor * 255).byte())
            img_path = os.path.join(img_dir, f"adv_{idx:04d}.png")
            img_pil.save(img_path)
            
            # Save mask
            if y_mask.dim() == 3:
                y_mask = y_mask.squeeze(0)
            mask_pil = to_pil((y_mask * 255).byte().unsqueeze(0))
            mask_path = os.path.join(mask_dir, f"mask_{idx:04d}.png")
            mask_pil.save(mask_path)
            
            saved_count += 1
            
        except Exception as e:
            logging.warning(f"Failed to save sample {idx}: {str(e)}")
            continue
    
    logging.info(f"Successfully saved {saved_count} adversarial samples")
    return saved_count


def load_adv_samples(load_dir, device='cpu'):
    """
    Load adversarial samples from disk.
    
    Args:
        load_dir: Directory path containing saved samples
        device: Device to load tensors to
    
    Returns:
        List of tuples (x_adv, y_mask) as tensors
    """
    img_dir = os.path.join(load_dir, "images")
    mask_dir = os.path.join(load_dir, "masks")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.warning(f"Sample directories not found in {load_dir}")
        return []
    
    samples = []
    to_tensor = transforms.ToTensor()
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for img_file in img_files:
        try:
            img_path = os.path.join(img_dir, img_file)
            idx = img_file.split('_')[1].split('.')[0]
            mask_path = os.path.join(mask_dir, f"mask_{idx}.png")
            
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            img_tensor = to_tensor(img).to(device)
            mask_tensor = to_tensor(mask).to(device)
            
            samples.append((img_tensor, mask_tensor))
            
        except Exception as e:
            logging.warning(f"Failed to load {img_file}: {str(e)}")
            continue
    
    logging.info(f"Loaded {len(samples)} adversarial samples from {load_dir}")
    return samples
