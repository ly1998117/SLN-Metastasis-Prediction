import torch
import os
import pandas as pd
import numpy as np
import lightning.pytorch as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImage, EnsureChannelFirst, Resize, Compose
from torchmetrics import MetricCollection, AUROC, Precision, Recall, Accuracy
from tqdm import tqdm
from mammoclip.image_embedding import get_encoder
from utils.logger import JsonLogs
from dataset.dataloader import DataHandler
from monai.utils import set_determinism

torch.multiprocessing.set_sharing_strategy('file_system')

from clip.model import Transformer, LayerNorm


class FeatDataset(Dataset):
    def __init__(self, datadir, transform=None):
        if isinstance(datadir, str):
            self.datadir = datadir
            self.data = pd.read_csv(os.path.join(datadir, 'CSV', 'data.csv')).to_dict(orient='records')
        else:
            self.data = datadir
            self.data['image'] = self.data['image'].map(
                lambda x: x.replace('image', 'clipfeat').replace('nii.gz', 'npy'))
            self.data = self.data.to_dict(orient='records')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item].copy()
        if self.transform:
            image_path = os.path.join(os.path.dirname(os.path.dirname(self.datadir)), item['path_image'])
            image = self.transform(image_path)
            image = image.permute(3, 0, 2, 1).repeat(1, 3, 1, 1)
            image -= image.min()
            image /= image.max()
            image = (image - 0.3089279) / 0.25053555408335154
            return image_path, image
        image = np.load(item['image'])
        label = item['label']
        return image, label


class FeatTransformer(torch.nn.Module):
    def __init__(self, length, input_dim, width, layers, heads, output_dim):
        super().__init__()
        self.pre_proj = torch.nn.Parameter(input_dim ** -0.5 * torch.randn(input_dim, width))
        scale = width ** -0.5
        self.class_embedding = torch.nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = torch.nn.Parameter(scale * torch.randn(length + 1, width))
        self.transformer = Transformer(width, layers, heads)
        self.ln_pre = LayerNorm(width)
        self.ln_post = LayerNorm(width)
        self.post_proj = torch.nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # shape = [B, L, width]
        x = x @ self.pre_proj
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        x = x @ self.post_proj
        return x


class FeatModel(L.LightningModule):
    def __init__(self, lr, loss_fn=torch.nn.CrossEntropyLoss(), length=128, width=768, layers=2, heads=12,
                 output_dim=512):
        super().__init__()
        self.loss_fn = loss_fn
        self.lr = lr
        self.encoder = FeatTransformer(length, 2048, width, layers, heads, output_dim)
        self.cls = torch.nn.Linear(output_dim, 2)
        self.train_metrics = MetricCollection([
            AUROC(task="multiclass", num_classes=2, average="weighted"),
            Accuracy(task="multiclass", num_classes=2, average="weighted"),
            Precision(task="multiclass", num_classes=2, average="weighted"),
            Recall(task="multiclass", num_classes=2, average="weighted")
        ], prefix='train_')
        self.val_metrics = self.train_metrics.clone(prefix="valid_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x):
        x = self.encoder(x)
        x = self.cls(x)
        return x

    def on_step_end_log(self, stage, preds, labels):
        preds = preds.softmax(dim=-1)
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
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        self.on_step_end_log('train', preds, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        preds = self(images)
        val_loss = self.loss_fn(preds, labels)
        # Update the validation metrics
        self.on_step_end_log('val', preds, labels)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.cls.parameters()), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        images, labels = batch
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


@torch.no_grad()
def extracting(dirpath='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped', device='cpu'):
    dataset = FeatDataset(dirpath, Compose([LoadImage(),
                                            EnsureChannelFirst(channel_dim='no_channel'),
                                            Resize(spatial_size=(512, 512, 128))]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    clip = get_encoder().to(device)

    for paths, image in tqdm(loader, desc='Extracting Features'):
        image = image.to(device)
        B, D, C, H, W = image.shape
        batch_features = clip(image.reshape(B * D, C, H, W)).reshape(B, D, -1).cpu().numpy()
        for path, feature in zip(paths, batch_features):
            path = path.replace('image', 'clipfeat').replace('nii.gz', 'npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, feature)
    data = pd.read_csv(os.path.join(dirpath, 'CSV', 'data.csv'))


def main(args):
    # set root log level to INFO and init a train logger, will be used in `StatsHandler`
    # setup_logging(log_file=f"{args.logs}/train.log", level=logging.INFO)
    os.makedirs(args.logs, exist_ok=True)
    JsonLogs(dir_path=args.logs, file_name='args.json')(args)

    set_determinism(seed=args.seed)

    # create UNet, DiceLoss and Adam optimizer
    model = FeatModel(lr=args.lr)

    data_handler = DataHandler(filepath=args.filepath,
                               column_renames=args.column_renames,
                               no_test=True,
                               postfix=args.postfix,
                               multi_label=args.multi_label,
                               select_labels=args.select_labels,
                               n_fold=args.n_fold,
                               n_repeat=args.n_repeat,
                               exclude_file=args.exclude_file,
                               )
    data = data_handler.get_split(repeat=args.repeat, fold=args.fold)
    train_loader = DataLoader(FeatDataset(data['train']), batch_size=args.batch_size, num_workers=args.workers, )
    val_loader = DataLoader(FeatDataset(data['val']), batch_size=args.batch_size, num_workers=args.workers)

    trainer = L.Trainer(
        devices=[args.device],
        max_epochs=args.epochs,
        logger=TensorBoardLogger(args.logs, name="training_logs"),
        callbacks=ModelCheckpoint(monitor='valid_MulticlassAUROC', dirpath=args.exp_path, filename='best_model',
                                  save_top_k=2, mode='max'),

    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    # extracting('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped', 1)
    from config import parse_args

    args = parse_args()

    main(args)
