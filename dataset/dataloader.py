import os
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from monai.data import CSVDataset, DataLoader
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ResizeWithPadOrCrop, Orientationd,
                              ScaleIntensityd, RandGaussianNoised, RandZoomd,
                              Resized, RandShiftIntensityd, NormalizeIntensityd, EnsureTyped, RandScaleIntensityd,
                              RandFlipd, ResizeWithPadOrCropd)
from .preprocess import N4BiasFieldCorrectiond
from utils.logger import PrintColor
from sklearn.utils import check_random_state


class RepeatFold:
    def __init__(self, n_repeat, n_fold, random_state=None):
        self.n_repeat = n_repeat
        self.n_fold = n_fold
        self.random_state = random_state

    def split(self, df: pd.DataFrame, shuffle=False, random_state=None):
        def _split(x):
            x.loc[:, 'fold'] = (list(range(self.n_fold)) * (len(x) // self.n_fold + 1))[:len(x)]
            return x

        if shuffle:
            df = df.sample(frac=1, random_state=check_random_state(random_state)).reset_index(drop=True)
        df = df.groupby('label').apply(_split)
        return df

    def repeat(self, df):
        rng = check_random_state(self.random_state)
        rps = []
        for nr in range(self.n_repeat):
            fold = self.split(df, shuffle=True, random_state=rng)
            fold['repeat'] = nr
            rps.append(fold)
        return pd.concat(rps)


def get_transforms(norm=False, spatial_size=(128, 128, 64)):
    train_transforms = [
        LoadImaged(keys=['image', 'mask'], image_only=False),
        EnsureChannelFirstd(['image', 'mask'], channel_dim='no_channel'),
        Orientationd(['image', 'mask'], axcodes='RAS'),
    ]
    test_transforms = [
        LoadImaged(keys=['image', 'mask'], image_only=False),
        EnsureChannelFirstd(['image', 'mask'], channel_dim='no_channel'),
        Orientationd(['image', 'mask'], axcodes='RAS')
    ]
    if isinstance(spatial_size, int):
        train_transforms.extend([
            Resized(['image', 'mask'], spatial_size=spatial_size, size_mode='longest'),
            # N4BiasFieldCorrectiond(['image']),
            ResizeWithPadOrCropd(['image', 'mask'], spatial_size=spatial_size),
            RandGaussianNoised(['image', 'mask'], prob=0.1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            RandZoomd(['image', 'mask'], prob=0.1),
        ])
        test_transforms.extend([
            Resized(['image', 'mask'], spatial_size=spatial_size, size_mode='longest'),
            # N4BiasFieldCorrectiond(['image']),
            ResizeWithPadOrCropd(['image', 'mask'], spatial_size=spatial_size),
        ])
    else:
        train_transforms.extend([
            Resized(['image', 'mask'], spatial_size=spatial_size),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        ])
        test_transforms.append(Resized(['image', 'mask'], spatial_size=spatial_size))
    if norm:
        train_transforms.append(NormalizeIntensityd(['image']))
        test_transforms.append(NormalizeIntensityd(['image']))
    else:
        train_transforms.append(ScaleIntensityd(['image']))
        test_transforms.append(ScaleIntensityd(['image']))
    train_transforms.append(RandScaleIntensityd(['image'], prob=0.5, factors=0.1))
    train_transforms.append(RandShiftIntensityd(['image'], prob=0.5, offsets=0.1))
    train_transforms.append(EnsureTyped(keys=['image', 'mask'], dtype=[torch.float, torch.long]))
    test_transforms.append(EnsureTyped(keys=['image', 'mask'], dtype=[torch.float, torch.long]))
    return Compose(train_transforms), Compose(test_transforms)


class DataHandler:
    def __init__(self, root_dir=None, output_dir=None, filename=None, filepath=None,
                 column_renames=None, postfix='', multi_label=False, n_repeat=1, n_fold=5,
                 no_test=False, select_labels: list = None, exclude_file=None):
        if filepath is not None:
            filename = os.path.basename(filepath)
            output_dir = os.path.basename(os.path.dirname(filepath))
            root_dir = os.path.dirname(os.path.dirname(filepath))
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.filename = filename
        self.five_fold = False
        self.n_fold = n_fold
        self.n_repeat = n_repeat
        self.column_renames = column_renames
        self.postfix = postfix
        self.multi_label = multi_label
        self.no_test = no_test
        self.select_labels = select_labels
        self.exclude_df = pd.read_csv(exclude_file) if exclude_file is not None else None
        self.data = None

    def _get_path(self, name, re_path=True):
        if re_path:
            name, ext = os.path.splitext(name)
            if self.n_repeat:
                name = f'{name}-repeat_{self.n_repeat}'
            if self.five_fold:
                name = f'{name}-{self.n_fold}_fold'
            if self.multi_label:
                name = f'{name}-multi_label'
            if self.no_test:
                name = f'{name}-no_test'
            if self.exclude_df is not None:
                name = f'{name}-exclude_{len(self.exclude_df)}'
            if self.postfix != '':
                name = f'{name}-{self.postfix}'
            name = f'{name}{ext}'

        if '.' not in name:
            name = f'{name}.csv'
        return os.path.join(self.root_dir, self.output_dir, name)

    def _get_df(self, name, df=None, re_path=True, rename=False, exclude_df=False):
        path = self._get_path(name, re_path)
        if df is not None:
            PrintColor.print(f'Save DataFrame to {path}', 'green')
            df.to_csv(path, index=False)
        elif os.path.exists(path):
            PrintColor.print(f'Load DataFrame from {path}', 'green')
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                raise ValueError(f"File format not supported: {path}")
        if df is None:
            df = False
        else:
            def set_data_fn(x):
                return os.path.join('dataset', x)

            if rename:
                df = df.rename(columns=self.column_renames)
                df['image'] = df['image'].map(set_data_fn)
                df['mask'] = df['mask'].map(set_data_fn)
            if self.exclude_df is not None and exclude_df:
                if 'image' in self.exclude_df.columns:
                    df = df[~df['image'].isin(self.exclude_df['image'])]
                else:
                    df = df[~df['image'].isin(self.exclude_df['path_image'])]
        return df

    def get_split(self, repeat=1, fold=None):
        if isinstance(fold, int) and fold >= 0:
            self.five_fold = True
        train = self._get_df('train')
        test = self._get_df('test')
        df = self._get_df(self.filename, re_path=False, rename=True, exclude_df=True)
        if self.select_labels is not None:
            df = df[df['label'].isin(self.select_labels)]
            df['label'] = df['label'].map(lambda x: self.select_labels.index(x))
        if self.no_test:
            test = None

        if test is False:
            test = df.groupby('label').sample(frac=0.2)
            test = self._get_df('test', test.reset_index(drop=True), re_path=False)

        if train is False:
            if self.no_test:
                train = df
            else:
                train = df[~df['id'].isin(test['id'])]
            train = self._get_df('train', train.reset_index(drop=True))

        if self.five_fold:
            folds = self._get_df('folds')

            if folds is False:
                train['fold'] = None
                folds = self._get_df('folds', RepeatFold(n_repeat=self.n_repeat, n_fold=self.n_fold).repeat(train))
            folds = folds[folds['repeat'] == repeat]
            train = folds[folds['fold'] != fold].drop(columns='fold').reset_index(drop=True)
            val = folds[folds['fold'] == fold].drop(columns='fold').reset_index(drop=True)
        else:
            val = None
        if self.column_renames:
            train = train[['image', 'mask', 'label']]
            test = test[['image', 'mask', 'label']] if test is not None else None
            if val is not None:
                val = val[['image', 'mask', 'label']]
        return dict(train=train, val=val, test=test)

    def _resampler(self, df):
        class_sample_count = df['label'].value_counts()
        weights = class_sample_count.sum() / class_sample_count
        samples_weights = weights[df['label']].tolist()
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
        return sampler

    def get_dataset(self, repeat=1, fold=None, spatial_size=(128, 128, 64), norm=False):
        _set_fn = lambda x, transform: CSVDataset(x,
                                                  column_names=['image', 'mask', 'label'],
                                                  transform=transform) if x is not None else None
        self.data = self.get_split(repeat, fold)
        train_transforms, test_transforms = get_transforms(norm=norm, spatial_size=spatial_size)
        train = _set_fn(self.data['train'], transform=train_transforms)
        val = _set_fn(self.data['val'], test_transforms)
        test = _set_fn(self.data['test'], test_transforms)
        return train, val, test

    def get_dataloader(self, repeat=1, fold=None, batch_size=1, workers=0, unbalanced=False,
                       spatial_size=(128, 128, 64), drop_last=False,
                       norm=False):
        _loader_fn = lambda x: DataLoader(x,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=drop_last,
                                          num_workers=workers) if x is not None else None
        train, val, test = self.get_dataset(repeat, fold, spatial_size, norm)
        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=not unbalanced,
                                  num_workers=workers,
                                  drop_last=drop_last,
                                  sampler=self._resampler(self.data['train']) if unbalanced else None)
        val_loader = _loader_fn(val)
        test_loader = _loader_fn(test)
        return dict(train=train_loader, val=val_loader, test=test_loader)
