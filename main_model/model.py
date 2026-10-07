from .unet import UNet
from .unet_plusplus import UNetPlusPlus
from .unet_attention import AttentionUNet

def init_model(model_type="Unet"):
    if model_type.lower() == "unet":
        return UNet()
    elif model_type.lower() == "unet++":
        return UNetPlusPlus()
    elif model_type.lower() == "unet++attention":
        return AttentionUNet()

    raise ValueError(
        f"Unknown model type: {model_type}. Supported model_type values: Unet, Unet++, Unet++Attention"
    )
