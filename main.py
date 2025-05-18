import logging
import os

import torch
import torch.nn as nn
import monai
import lightning.pytorch as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from monai.apps import get_logger
from monai.utils import set_determinism

from torchmetrics import Accuracy, Recall, Precision, AUROC
from torchmetrics.collections import MetricCollection
from utils.logger import JsonLogs
from dataset.dataloader import DataHandler
from models import get_model

torch.multiprocessing.set_sharing_strategy('file_system')


class Model(L.LightningModule):
    def __init__(self, model_name, lr, epochs, loss_fn=nn.CrossEntropyLoss(), act=nn.Softmax(dim=-1)):
        super(Model, self).__init__()
        self.net = get_model(model_name, 2, 1, False)
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        self.act = act
        self.train_metrics = MetricCollection([
            AUROC(task="multiclass", num_classes=2, average="macro"),
            Accuracy(task="multiclass", num_classes=2, average="macro"),
            Precision(task="multiclass", num_classes=2, average="macro"),
            Recall(task="multiclass", num_classes=2, average="macro")
        ], prefix='train_')
        self.val_metrics = self.train_metrics.clone(prefix="valid_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x):
        return self.net(x)

    def on_step_end_log(self, stage, preds, labels):
        preds = self.act(preds)
        if stage == 'train':
            metrics = self.train_metrics
        elif stage == 'val':
            metrics = self.val_metrics
        else:
            metrics = self.test_metrics
        metrics.update(preds, labels)
        if stage == 'train':
            self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        images = images.as_tensor()
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.on_step_end_log('train', preds, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        images = images.as_tensor()

        preds = self(images)
        val_loss = self.loss_fn(preds, labels)
        # Update the validation metrics
        self.on_step_end_log('val', preds, labels)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                      lr_lambda=lambda epoch: (1 - epoch / self.epochs) ** 0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Update every epoch
                'frequency': 1,  # Every epoch
            }
        }

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        images = images.as_tensor()
        preds = self.forward(images)
        test_loss = self.loss_fn(preds, labels)

        self.on_step_end_log('test', preds, labels)
        return test_loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        # Log the final validation metrics at the end of the epoch
        self.log_dict(self.val_metrics.compute(), on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        # Log the final test metrics at the end of the testing phase
        self.log_dict(self.test_metrics.compute(), on_epoch=True, logger=True)
        self.test_metrics.reset()


def main(args):
    # set root log level to INFO and init a train logger, will be used in `StatsHandler`
    # setup_logging(log_file=f"{args.logs}/train.log", level=logging.INFO)
    os.makedirs(args.logs, exist_ok=True)
    JsonLogs(dir_path=args.logs, file_name='args.json')(args)
    get_logger("train_log", logger_handler=logging.FileHandler(f"{args.logs}/train.log"))

    set_determinism(seed=args.seed)
    monai.config.print_config()

    # create UNet, DiceLoss and Adam optimizer
    model = Model(model_name=args.model_name, lr=args.lr, epochs=args.epochs)

    data_handler = DataHandler(filepath=args.filepath,
                               column_renames=args.column_renames,
                               no_test=args.no_test,
                               postfix=args.postfix,
                               multi_label=args.multi_label,
                               select_labels=args.select_labels,
                               n_fold=args.n_fold,
                               n_repeat=args.n_repeat,
                               exclude_file=args.exclude_file,
                               )
    dataloader = data_handler.get_dataloader(repeat=args.repeat,
                                             fold=args.fold,
                                             batch_size=args.batch_size,
                                             workers=args.workers,
                                             unbalanced=args.unbalanced,
                                             spatial_size=args.spatial_size,
                                             norm=args.norm)
    train_loader, val_loader, test_loader = dataloader['train'], dataloader['val'], dataloader['test']
    trainer = L.Trainer(
        devices=[args.device],
        max_epochs=args.epochs,
        logger=TensorBoardLogger(args.logs, name="training_logs"),
        callbacks=ModelCheckpoint(monitor='valid_MulticlassAUROC', dirpath=args.exp_path, filename='best_model',
                                  save_top_k=2,
                                  mode='min'),

    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    from config import parse_args

    args = parse_args()
    # args.name = 'Debug'
    # args.filepath = filepath='dataset/data/CSV_CROPPED/data.csv'
    # args.spatial_size = (96, 96, 96)
    # args.no_test = True
    main(args)
