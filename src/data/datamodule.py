from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torchvision.transforms import transforms
from .dataset import CustomDataset
from torch.utils.data import DataLoader, Dataset
from src.data.components.augmentation import get_training_augmentation

class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 1,
        Labels: dict = None,
        SIZE_IMAGE: tuple = (448,448),
        path_annotation_train: str = "",
        path_annotation_val: str = "",
        path_annotation_test: str = "",
        path_images_data: str = "",
        priority: tuple = (),
        augmentation_path: str = "../configs/data/augmentation.yaml"

    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.data_train = CustomDataset(
                path_annotation_data=self.hparams.path_annotation_train,
                path_image_data=self.hparams.path_images_data,
                label=self.hparams.Labels,
                size=self.hparams.SIZE_IMAGE,
                augmentation=get_training_augmentation(self.hparams.augmentation_path),
                priority = self.hparams.priority
            )
            self.data_val = CustomDataset(
                path_annotation_data=self.hparams.path_annotation_val,
                path_image_data=self.hparams.path_images_data,
                label=self.hparams.Labels,
                size=self.hparams.SIZE_IMAGE,
                augmentation=None,
                priority = self.hparams.priority
            )

        elif stage == 'test':
            self.data_test = CustomDataset(
                path_annotation_data=self.hparams.path_annotation_test,
                path_image_data=self.hparams.path_images_data,
                label=self.hparams.Labels,
                size=self.hparams.SIZE_IMAGE,
                augmentation=None,
                priority=self.hparams.priority
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModule()
