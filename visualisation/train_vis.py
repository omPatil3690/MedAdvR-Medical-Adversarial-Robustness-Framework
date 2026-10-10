import torch
import matplotlib.pyplot as plt

def visualize_predictions(model, loader, device, num_samples=3):
    model.eval()
    shown = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.4).float()

            imgs = imgs.cpu()
            masks = masks.cpu()
            preds = preds.cpu()

            for i in range(imgs.shape[0]):
                if shown >= num_samples:
                    return

                img = imgs[i].permute(1, 2, 0)
                mask = masks[i][0]
                pred = preds[i][0]

                plt.figure(figsize=(10, 3))

                plt.subplot(1, 3, 1)
                plt.title("Image")
                plt.imshow(img)
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title("GT Mask")
                plt.imshow(mask, cmap="gray")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title("Prediction")
                plt.imshow(pred, cmap="gray")
                plt.axis("off")

                plt.show()
                shown += 1
