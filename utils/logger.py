import argparse
import os
import json
import logging
import time
import pandas as pd
import numpy as np
from ast import literal_eval


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


class Logger:
    def __init__(self, dir_path, file_name='logs'):
        self.dir_path = dir_path
        self.file_name = file_name
        self.logs = []
        self.path = os.path.join(dir_path, f'{file_name}.txt')

    def log(self, text):
        self.logs.append(text)
        with open(self.path, 'a+') as f:
            print(text, file=f)

    def __call__(self, text):
        self.log(text)

    def __str__(self):
        return '\n'.join(self.logs)

    def __repr__(self):
        return self.__str__()


class Logs:
    def __init__(self, dir_path, file_name='logs'):
        dir_path = os.path.join(dir_path, 'TXTLogger')
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, f'{file_name}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.txt')

    def log(self, text):
        with open(self.path, 'a+') as f:
            print(text, file=f)

    def __call__(self, t):
        self.log(t)


class TensorboardLogs:
    def __init__(self, dir_path):
        dir_path = os.path.join(dir_path, 'TensorboardLogger')
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(dir_path)

    def add_image(self):
        pass


class JsonLogs:
    def __init__(self, dir_path=None, file_name='args.json', file_path=None):
        if file_path is not None:
            dir_path, file_name = os.path.split(file_path)
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, f'{file_name}')

    def txt2json(self):
        with open(self.path, 'r') as f:
            lines = f.read().splitlines()
        args_dict = {}
        for line in lines:
            if ('<' in line or '>' in line or ': ' not in line or
                    '(' in line or ')' in line or '[' in line or ']' in line):
                continue
            k, v = line.split(': ')
            if v.isdigit():
                v = int(v)
            elif v.replace('.', '', 1).isdigit():
                v = float(v)
            elif v in ['True', 'False']:
                v = True if v == 'True' else False
            elif v == 'None':
                v = None
            args_dict[k] = v
        return args_dict

    def log(self, args):
        # 将 args 转换为字典
        args_dict = {k: v for k, v in vars(args).items() if isinstance(v, (int, str, float, bool))}

        # 将字典保存为 JSON 文件
        with open(self.path, "w") as f:
            json.dump(args_dict, f, indent=4)

    def read(self, args=None, ignore_keys=None):
        if os.path.exists(self.path) and self.path.endswith('.json'):
            with open(self.path, 'r') as f:
                args_dict = json.load(f)
        else:
            args_dict = self.txt2json()
        if args is None:
            args = argparse.Namespace()
        for k, v in args_dict.items():
            if ignore_keys is not None and k in ignore_keys and hasattr(args, k):
                continue
            setattr(args, k, v)
        return args

    def __call__(self, args):
        self.log(args)


class MarkDownLogs:
    def __init__(self, dir_path):
        self.dir_path = os.path.join(dir_path, 'MDLogger')
        os.makedirs(self.dir_path, exist_ok=True)

    def __call__(self, text, file_name):
        path = os.path.join(self.dir_path, f'{file_name}.md')
        with open(path, 'w') as f:
            f.write(text)


class CSVLogs:
    def __init__(self, dir_path=None, logger=None, file_name='logs', overwrite=False):
        if dir_path is None:
            self.dir_path = logger.dir_path
        else:
            self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)
        self.df = pd.DataFrame()
        self.path = os.path.join(self.dir_path, file_name)
        self.overwrite = overwrite

    def filepath(self, filepath):
        self.path = filepath

    def filename(self, filename, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path
        self.path = os.path.join(dir_path, filename)

    def cache(self, text):
        try:
            if isinstance(text, dict):
                text = pd.DataFrame([text])
            elif isinstance(text, list):
                text = pd.DataFrame(text)
            elif isinstance(text, pd.DataFrame):
                pass
            else:
                raise Exception('text type error')
            self.df = pd.concat([self.df, text], axis=0).reset_index(drop=True)
        except:
            import pdb
            pdb.set_trace()

    def write(self, save_key=None):
        if os.path.exists(self.path) and not self.overwrite:
            df = pd.read_csv(self.path)
            self.df = pd.concat([df, self.df], axis=0).reset_index(drop=True)
        if save_key is not None:
            value = self.df[save_key].unique()[0]
            path = self.path.replace('.csv', f'_{value}.csv')
        else:
            path = self.path
        self.df.to_csv(path, index=False)

    def __call__(self, text):
        self.cache(text)
        self.write()

    def dataframe(self):
        if len(self.df) == 0:
            df = pd.read_csv(self.path)
            for c in df.columns:
                try:
                    df[c] = df[c].map(lambda x: literal_eval(x))
                except Exception:
                    pass
            self.df = df
        return self.df

    def _output_error_name(self, df):
        names = df['names'].tolist()[0]
        pred = np.array(df['scores'].to_list()).squeeze().argmax(-1).tolist()
        labels = df['labels'].tolist()[0]
        df = pd.DataFrame({'name': names, 'label': labels, 'pred': pred})
        errors = df[df['label'] != df['pred']]
        return errors

    def output_error_name(self):
        df = self.dataframe()
        df = df.groupby(['epoch', 'stage_name', 'modality']).apply(self._output_error_name).reset_index()[
            ['epoch', 'stage_name', 'modality', 'name', 'label', 'pred']]
        df.to_csv(os.path.join(self.dir_path, f'error_name.csv'), index=False)
        return df

    def preds_and_labels(self, stage_name, epoch, modality):
        df = self.dataframe()
        raw = df[(df['stage_name'] == stage_name) & (df['epoch'] == epoch) & (df['modality'] == modality)][
            ['labels', 'scores']].iloc[0]
        preds = np.array(raw['scores'])[:, :3].argmax(-1)
        labels = np.array(raw['labels'])
        return preds, labels


def char_color(s, front=50, color='green'):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    color_codes = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }
    color_code = color_codes.get(color.lower(), '32')  # 默认为绿色
    new_char = "\033[0;" + color_code + ";" + str(int(front)) + "m" + s + "\033[0m"
    return new_char


def make_dirs(path):
    try:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, mode=0o777)
    except Exception:
        pass


def deepsearch(path, op, n=0):
    if not os.path.exists(path):
        return
    op(path, n)
    if os.path.isfile(path):
        return
    try:
        for sub in os.listdir(path):
            deepsearch(os.path.join(path, sub), op, n + 1)
    except:
        print('ERROR: ', path)
        return


class CharColor:
    color_codes = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }

    @classmethod
    def char(cls, s, front=50, color='green'):
        color_code = cls.color_codes.get(color.lower(), '32')  # 默认为绿色
        new_char = "\033[0;" + color_code + ";" + str(int(front)) + "m" + s + "\033[0m"
        return new_char


class PrintColor(CharColor):
    def __init__(self, s, color='green', front=50):
        PrintColor.print(s, front=front, color=color)

    @classmethod
    def print(cls, s, color='green', front=50):
        print(super().char(s, front=front, color=color))
