import torch
from torch import nn
import segmentation_models_pytorch as smp


class unet(nn.Module):

    def __init__(
            self,
            ENCODER: str,
            ENCODER_WEIGHTS: str,
            NUM_CLASSES: int,
            ACTIVATION: str
    ) -> None:
        super().__init__()

        self.model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=NUM_CLASSES,
            activation=ACTIVATION,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    _ = unet()
