from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import JaccardIndex
from monai.networks.utils import one_hot
import segmentation_models_pytorch

class Module(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        criterion1: torch.nn,
        criterion2: None,
        weight_criterion1: float,
        weight_criterion2: float,
        num_classes: int,
        task: str
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.weight_criterion1 = weight_criterion1
        self.weight_criterion2 = weight_criterion2
        self.num_classes = num_classes

        self.train_iou_metric = JaccardIndex(num_classes=num_classes, task = task, average=None)
        self.val_iou_metric = JaccardIndex(num_classes=num_classes, task = task, average=None)
        self.test_iou_metric = JaccardIndex(num_classes=num_classes, task = task, average=None)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation f1
        self.val_mean_IOU_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        preds = self.forward(x)

        loss = (self.criterion1(preds, y.long()) * self.weight_criterion1 +
                    self.criterion2(preds, y.long()) * self.weight_criterion2)


        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_iou_metric(preds.argmax(dim=1), targets.squeeze(1))

        # self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        average_iou_train = self.train_iou_metric.compute()
        mean_iou_train = torch.mean(average_iou_train).item()
        print(average_iou_train.shape)
        self.log("train/mean_IOU", mean_iou_train, prog_bar=True)
        self.log("train/lane_IOU", average_iou_train[0].item(), prog_bar=True)
        self.log("train/border_IOU", average_iou_train[1].item(), prog_bar=True)
        self.log("train/road_IOU", average_iou_train[2].item(), prog_bar=True)
        self.log("train/vehicle_IOU", average_iou_train[3].item(), prog_bar=True)
        self.log("train/ego_car_IOU", average_iou_train[4].item(), prog_bar=True)
        self.log("train/puddle_IOU", average_iou_train[5].item(), prog_bar=True)
        self.log("train/snow_IOU", average_iou_train[6].item(), prog_bar=True)

        self.train_iou_metric.reset()


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_iou_metric(preds.argmax(dim=1), targets.squeeze(1))


        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        average_iou_val = self.val_iou_metric.compute()
        mean_iou_val = torch.mean(average_iou_val).item()

        self.val_mean_IOU_best(mean_iou_val)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mean_IOU_best", self.val_mean_IOU_best.compute(), sync_dist=True, prog_bar=True)

        self.log("val/mean_IOU", mean_iou_val, prog_bar=True)
        self.log("val/lane_IOU", average_iou_val[0].item(), prog_bar=True)
        self.log("val/border_IOU", average_iou_val[1].item(), prog_bar=True)
        self.log("val/road_IOU", average_iou_val[2].item(), prog_bar=True)
        self.log("val/vehicle_IOU", average_iou_val[3].item(), prog_bar=True)
        self.log("val/ego_car_IOU", average_iou_val[4].item(), prog_bar=True)
        self.log("val/puddle_IOU", average_iou_val[5].item(), prog_bar=True)
        self.log("val/snow_IOU", average_iou_val[6].item(), prog_bar=True)

        self.val_iou_metric.reset()


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_iou_metric(preds.argmax(dim=1), targets.squeeze(1))

        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        average_iou_test = self.test_iou_metric.compute()
        mean_iou_test = torch.mean(average_iou_test).item()

        self.log("test/mean_IOU", mean_iou_test, prog_bar=True)
        self.log("test/lane_IOU", average_iou_test[0].item(), prog_bar=True)
        self.log("test/border_IOU", average_iou_test[1].item(), prog_bar=True)
        self.log("test/road_IOU", average_iou_test[2].item(), prog_bar=True)
        self.log("test/vehicle_IOU", average_iou_test[3].item(), prog_bar=True)
        self.log("test/ego_car_IOU", average_iou_test[4].item(), prog_bar=True)
        self.log("test/puddle_IOU", average_iou_test[5].item(), prog_bar=True)
        self.log("test/snow_IOU", average_iou_test[6].item(), prog_bar=True)


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mean_IOU_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Module(None, None, None, None)
