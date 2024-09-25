import numpy as np
import torch
import albumentations as albu
import clearml

from lightning import Callback, Trainer
from src.models.module import Module
from typing import Tuple
from torch import Tensor
from torchvision.utils import make_grid
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class VisualizePredMask(Callback):
    def __init__(self, every_n_epochs: int, color_mapping: dict, size: Tuple, grid_row: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.color_mapping = color_mapping
        self.size = tuple(map(int, size.strip('()').split(',')))
        self.grid_row = grid_row

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:  # noqa: WPS210
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        images, gt_masks = next(iter(trainer.train_dataloader))

        visualizations = []
        for img in images:
            img = denormalize(tensor_to_cv_image(img))
            visualizations.append(cv_image_to_tensor(img, normalize=False))

        for gt_mask in gt_masks:
            mask_np = gt_mask.squeeze().numpy()
            mask_rgb = np.zeros(shape=self.size, dtype=np.uint8)
            for label, rgb_values in self.color_mapping.items():
                mask_label = (mask_np == int(label))
                mask_rgb[mask_label] = rgb_values
            visualizations.append(cv_image_to_tensor(mask_rgb, normalize=False))

        logits = pl_module(images.to(device=pl_module.device))
        for _, logit_m in enumerate(logits):
            logit_mask = torch.sigmoid(logit_m)
            mask = logit_mask.argmax(dim=0).to(int)
            mask_np = mask.squeeze().cpu().detach().numpy()
            mask_rgb = np.zeros(shape=self.size, dtype=np.uint8)

            for label, rgb_values in self.color_mapping.items():
                mask_label = (mask_np == int(label))
                mask_rgb[mask_label] = rgb_values

            visualizations.append(cv_image_to_tensor(mask_rgb, normalize=False))

        grid = make_grid(visualizations, nrow=self.grid_row, normalize=False)

        trainer.logger.experiment.add_image(
            'Mask preview',
            img_tensor=grid,
            global_step=trainer.global_step,
        )


def denormalize(
        img: NDArray[float],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        max_value: int = 255,
) -> NDArray[int]:
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)['image'] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray[float]:
    return tensor.permute(1, 2, 0).cpu().numpy()


def cv_image_to_tensor(img: NDArray[float], normalize: bool = True) -> Tensor:
    ops = [ToTensorV2()]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)['image']

