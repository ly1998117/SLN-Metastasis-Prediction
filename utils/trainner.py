# -*- coding: utf-8 -*-
"""
@Time : 2024/8/20 19:06 
@Author :   liuyang 
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
@File :     trainner.py 
"""
import torch
from monai.transforms import Compose, AsDiscreted, Activationsd, AsDiscrete
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    ROCAUC,
    EarlyStopHandler,
    LrScheduleHandler,
    MetricLogger,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from ignite.metrics import Accuracy, Recall, Precision
from utils.handlers import ClassificationSaver


def get_lr_schedular(opt, epochs, name=''):
    if name == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    elif name == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif name == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    else:
        raise KeyError
    return lr_scheduler


def get_evaluator(args, device, net, val_loader, test_loader, post_transforms, ):
    def post_fn(x):
        d = x[0]['pred'].device
        x = from_engine(["pred", "label"])(x)
        y = [AsDiscrete(to_onehot=args.num_classes)(i).to(d) for i in x[1]]
        return x[0], y

    def _eva(stage, loader):
        if loader is None:
            return loader
        handlers = [
            StatsHandler(name="train_log", tag_name=stage, output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=f"{args.exp_path}/tensorboard", output_transform=lambda x: None),
            # TensorBoardImageHandler(
            #     log_dir=f"{args.exp_path}/tensorboard",
            #     batch_transform=from_engine(["image"]),
            #     output_transform=from_engine(["pred"]),
            # ),
            CheckpointSaver(name="train_log", save_dir=f"{args.exp_path}/checkpoint", save_dict={"net": net},
                            file_prefix=stage, save_key_metric=True),
            ClassificationSaver(name="train_log",
                                output_dir=f"{args.exp_path}/classification",
                                filename=f'{stage}.csv',
                                batch_transform=from_engine("image_meta_dict"),
                                output_transform=from_engine(["pred", "label"]))]
        return SupervisedEvaluator(
            device=device,
            val_data_loader=val_loader,
            network=net,
            key_val_metric={
                f"{stage}_roc_auc": ROCAUC(output_transform=post_fn),
            },
            additional_metrics={
                f"{stage}_recall": Recall(output_transform=from_engine(["pred", "label"]), average=True),
                f"{stage}_precision": Precision(output_transform=from_engine(["pred", "label"]),
                                                average=True),
                f"{stage}_acc": Accuracy(output_transform=from_engine(["pred", "label"]))
            },
                val_handlers=handlers,
            postprocessing=post_transforms,
            amp=False,
        )

    val_evaluator = _eva('val', val_loader)
    test_evaluator = _eva('test', test_loader)
    return val_evaluator, test_evaluator