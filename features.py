# -*- coding: utf-8 -*-
"""
@Time : 2024/8/20 18:56
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
@File :     pretrain.py
"""
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
from models.SwinViT import SSLHead
from dataset import DataHandler
from torch.utils.data import DataLoader, Dataset


class BatchDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Projector(torch.nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(Projector, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class ConvProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(ConvProjector, self).__init__()
        self.model = torch.nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1)
        self.max_pooling = torch.nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        x = self.model(x)
        x = self.max_pooling(x).squeeze()
        return x


def tsne_plot(features, labels):
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, random_state=0)
    features = tsne.fit_transform(features)
    plt.figure(figsize=(10, 10))
    for i in range(2):
        plt.scatter(features[labels == i, 0], features[labels == i, 1], label=str(i))
    plt.legend()
    plt.show()


def get_features(args):
    model = SSLHead(
        in_channels=1,
        feature_size=48,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
    )
    postfix = 'old'
    model_dict = torch.load(f'/data_smr/liuy/Project/BreastCancer/pretrain/log_{postfix}/model_bestValRMSE.pt',
                            map_location='cpu')["state_dict"]
    model_dict = {k.replace("module.", ""): v for k, v in model_dict.items()}
    model.load_state_dict(model_dict)
    model.to(args.device)
    dataloader = DataHandler(filepath='dataset/data/CSV_BREAST/data.csv',
                             column_renames={'path_data': 'image',
                                             'SLN状态（0_无转移，1_转移）': 'label',
                                             'path_mask': 'mask'},
                             no_test=True,
                             ).get_dataloader(0,
                                              fold=0,
                                              batch_size=args.batch_size,
                                              workers=2,
                                              unbalanced=False,
                                              spatial_size=(192, 96, 96),
                                              norm=True)
    train_loader = dataloader['train']
    test_loader = dataloader['val']
    model.eval()
    if not os.path.exists(f"./pretrain/features{postfix}.pt"):
        features = []
        labels = []
        filepaths = []
        with torch.no_grad():
            for batch in tqdm(train_loader):
                x = batch["image"].to(args.device)
                y = batch["label"]
                for f_i, y_i in zip(x.meta['filename_or_obj'], y.cpu().tolist()):
                    filepaths.append({'path_data': f_i, 'label': y_i})
                feature = model.embedding(x).cpu()
                features.append(feature)
                labels.append(y)
            for batch in tqdm(test_loader):
                x = batch["image"].to(args.device)
                y = batch["label"]
                for f_i, y_i in zip(x.meta['filename_or_obj'], y.cpu().tolist()):
                    filepaths.append({'path_data': f_i, 'label': y_i})
                feature = model.embedding(x).cpu()
                features.append(feature)
                labels.append(y)
        filepaths = pd.DataFrame(filepaths)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        filepaths.to_csv(f"./pretrain/filepaths{postfix}.csv", index=False)
        torch.save(features, f"./pretrain/features{postfix}.pt")
        torch.save(labels, f"./pretrain/labels{postfix}.pt")


def find_not_similar(filedf, features, labels):
    class0 = features[labels == 0]
    class1 = features[labels == 1]
    class0_df = filedf.loc[filedf['label'] == 0].reset_index(drop=True)
    class1_df = filedf.loc[filedf['label'] == 1].reset_index(drop=True)
    class0_norm = class0.flatten(start_dim=1) / class0.flatten(start_dim=1).norm(dim=1, keepdim=True)
    class1_norm = class1.flatten(start_dim=1) / class1.flatten(start_dim=1).norm(dim=1, keepdim=True)
    sim = (class0_norm @ class1_norm.T).mean(1)  # N0
    thresh = sim.topk(int(len(sim) * 0.6))[0].min()
    selected = sim < thresh
    class0 = class0[selected]
    class0_dropped = class0_df.loc[~selected.numpy()].reset_index(drop=True)
    class0_df = class0_df.loc[selected.numpy()].reset_index(drop=True)
    labels0 = labels[labels == 0][selected]
    labels1 = labels[labels == 1]
    print(f"Selected Size: {class0.shape[0]} {class1.shape[0]} Thresh: {thresh}")
    class0_dropped.to_csv("./pretrain/class0_dropped.csv", index=False)
    return class0, class1, labels0, labels1


def train(args):
    filedf = pd.read_csv("./pretrain/filepaths.csv")
    features = torch.load("./pretrain/features.pt").as_tensor()
    labels = torch.load("./pretrain/labels.pt")
    class0, class1, labels0, labels1 = find_not_similar(filedf, features, labels)
    train_feature = torch.cat([class0[:int(0.8 * len(class0))], class1[:int(0.8 * len(class1))]], dim=0)
    train_label = torch.cat([labels0[:int(0.8 * len(class0))], labels1[:int(0.8 * len(class1))]], dim=0)
    test_feature = torch.cat([class0[int(0.8 * len(class0)):], class1[int(0.8 * len(class1)):]], dim=0)
    test_label = torch.cat([labels0[int(0.8 * len(class0)):], labels1[int(0.8 * len(class1)):]], dim=0)
    train_loader = DataLoader(BatchDataset(train_feature, train_label), batch_size=64, shuffle=True)
    test_loader = DataLoader(BatchDataset(test_feature, test_label), batch_size=64, shuffle=False)
    projector = ConvProjector(features.shape[1], features.shape[1])
    projector.to(args.device)
    projector.train()
    classifier = torch.nn.Linear(features.shape[1], 2)
    classifier.to(args.device)
    classifier.train()
    optimizer = torch.optim.Adam(list(projector.parameters()) + list(classifier.parameters()), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        acc = 0
        for batch in tqdm(train_loader):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            x = projector(x)
            y_pred = classifier(x)
            loss = criterion(y_pred, y)
            acc += (y_pred.argmax(dim=1) == y).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Train Epoch: {epoch}, Loss: {loss.item()} Acc: {acc / len(train_loader)}")
        acc = 0
        pred = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                x, y = batch
                x = x.to(args.device)
                y = y.to(args.device)
                x = projector(x)
                y_pred = classifier(x)
                pred.append(y_pred.argmax(dim=1))
                labels.append(y.cpu())
                acc += (y_pred.argmax(dim=1) == y).float().mean()
        print(f"Test Epoch: {epoch}, Acc: {acc / len(test_loader)}")
        # pred = torch.cat(pred, dim=0)
        # labels = torch.cat(labels, dim=0)
        # for p, l in zip(pred, labels):
        #     print(p, l)

    projector.eval()
    classifier.eval()
    dataloader = DataLoader(BatchDataset(
        torch.cat([class0, class1], dim=0),
        torch.cat([labels0, labels1], dim=0),
    ), batch_size=64, shuffle=False)
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x = x.to(args.device)
            x = projector(x)
            features.append(x.cpu())
            labels.append(y.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    tsne_plot(features, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="log", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")

    args = parser.parse_args()
    args.device = "cuda:0"

    get_features(args)
    train(args)
