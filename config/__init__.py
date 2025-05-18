import argparse
import ast
import os


def get_config(config_name):
    from .basic_config import get_basic_config
    basic, config_name = config_name.split('_')
    basic = get_basic_config(basic)
    if 'crop' == config_name:
        from .crop import config
        basic.update(config)
    elif 'breast' == config_name:
        from .breast import config
        basic.update(config)
    else:
        raise ValueError(f"Config {config_name} not supported")
    return basic


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args=None, debug_dict=None, add_config=True, json_path=None):
    parser = argparse.ArgumentParser()
    # Path specific arguments
    parser.add_argument(
        "--root_dir",
        type=str,
        default="runs",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--fold",
        type=lambda x: int(x) if x.isdigit() else x,
        default=0,
        help="Fold number."
    )
    parser.add_argument(
        "--n_fold",
        default=5,
        type=int,
        help="n_fold cross-validation"
    )
    parser.add_argument(
        "--repeat",
        type=lambda x: int(x) if x.isdigit() else x,
        default=0,
        help="Fold number."
    )
    parser.add_argument(
        "--n_repeat",
        default=1,
        type=int,
        help="n_fold cross-validation"
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="logs",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--no_test",
        default=False,
        action="store_true",
        help="If true, the test set is not initialized."
    )
    parser.add_argument(
        "--exclude_file",
        type=str,
        default=None,
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    # Training  specific arguments
    parser.add_argument(
        "--config",
        type=str,
        default="default_crop",
        help="Which config to use.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default='poly',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--save_frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=lambda x: int(x) if x.isdigit() else x,
        help="",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Default random seed."
    )

    # Dataset specific arguments
    parser.add_argument(
        "--unbalanced",
        default=False,
        action="store_true",
        help="If true, the dataloader will use a weighted sampler to handle class imbalance."
    )
    parser.add_argument(
        "--norm",
        default=False,
        action="store_true",
        help="If true, the dataloader will normalize the images."
    )
    # Model specific arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet-b0",
        help="Model to use. One of: 'resnet18', 'resnet34', 'efficient-b0', 'efficient-b1', 'efficient-b2', 'efficient-b3', 'efficient-b4', 'efficient-b5', 'efficient-b6', 'efficient-b7'.",
    )
    args = parser.parse_args(args)
    if json_path is not None:
        from utils.logger import JsonLogs
        args = JsonLogs(file_path=json_path).read(args, ignore_keys=['exp_path', 'logs'])

    if debug_dict is not None:
        args.__dict__.update(debug_dict)

    # If some params are not passed, we use the default values based on model name.
    if add_config:
        config = get_config(args.config)
        for name, val in config.items():
            setattr(args, name, val)
    args.exp_path = os.path.join(args.root_dir, args.name, f"repeat_{args.repeat}", f"fold_{args.fold}")
    args.logs = os.path.join(args.exp_path, args.logs)

    return args
