import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

def compute_ssim(x1, x2, data_range=1.0):
    """Compute SSIM for batch of images."""
    vals = []
    for i in range(x1.shape[0]):
        val = ssim(x1[i].unsqueeze(0), x2[i].unsqueeze(0), data_range=data_range)
        vals.append(val)
    return torch.stack(vals).mean()


def edge_loss(x_adv, x):
    """Specialized loss for edge generator: -|∇(x_adv) - ∇x|"""
    grad_adv = torch.gradient(x_adv, dim=[2, 3])
    grad_x = torch.gradient(x, dim=[2, 3])
    diff_y = torch.abs(grad_adv[0] - grad_x[0])
    diff_x = torch.abs(grad_adv[1] - grad_x[1])
    return -torch.mean(diff_y + diff_x)


def intensity_loss(x_adv, x):
    """Specialized loss for intensity generator: -|x - x_adv|"""
    return -torch.mean(torch.abs(x_adv - x))


def texture_loss(x_adv, x):
    """Specialized loss for texture generator: -|F(x) - F(x_adv)|"""
    f_x = torch.fft.fft2(x)
    f_x_adv = torch.fft.fft2(x_adv)
    return -torch.mean(torch.abs(f_x - f_x_adv))


def specialized_loss(x_adv, x, gen_type):
    """Compute specialized loss based on generator type."""
    if gen_type == 'edge':
        return edge_loss(x_adv, x)
    elif gen_type == 'intensity':
        return intensity_loss(x_adv, x)
    elif gen_type == 'texture':
        return texture_loss(x_adv, x)
    else:
        return torch.tensor(0.0, device=x.device)


def edge_diff_score(x1, x2):
    """Compute edge difference score."""
    grad1 = torch.gradient(x1, dim=[2, 3])
    grad2 = torch.gradient(x2, dim=[2, 3])
    diff = torch.abs(grad1[0] - grad2[0]) + torch.abs(grad1[1] - grad2[1])
    return torch.mean(diff)


def realism_score(x, x_adv, gen_type):
    """Compute realism score R(x, x_adv) based on generator type."""
    ssim_val = compute_ssim(x, x_adv)
    if gen_type == 'edge':
        intensity_dev = torch.mean(torch.abs(x_adv - x))
        return 0.5 * ssim_val + 0.5 * (1 - intensity_dev)
    elif gen_type == 'intensity':
        edge_diff = edge_diff_score(x, x_adv)
        return 0.5 * ssim_val + 0.5 * (1 - edge_diff)
    elif gen_type == 'texture':
        f_x = torch.fft.fft2(x)
        f_x_adv = torch.fft.fft2(x_adv)
        freq_dev = torch.mean(torch.abs(f_x - f_x_adv))
        return 0.5 * ssim_val + 0.5 * (1 - freq_dev)
    else:
        return ssim_val


def realism_loss(x, x_adv, gen_type, tau=0.9):
    """
    Convert realism score to loss using barrier function.
    
    Realism score R: 0 to 1 (higher = more realistic)
    Constraint: R >= tau
    Loss: relu(tau - R)
    
    When R >= tau: Loss = 0 (constraint satisfied)
    When R < tau:  Loss > 0 (constraint violated, penalty applied)
    
    Args:
        x: Original clean image
        x_adv: Adversarial perturbed image  
        gen_type: Type of generator ('edge', 'intensity', 'texture')
        tau: Realism threshold (default 0.9)
    
    Returns:
        Realism loss (barrier function penalty)
    """
    R = realism_score(x, x_adv, gen_type)
    L_real = torch.relu(tau - R)
    return L_real