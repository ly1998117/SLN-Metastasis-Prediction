# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""
import os
import signal
import subprocess
import concurrent.futures
from types import NoneType

from tqdm import tqdm


class MultiRunner:
    def __init__(self, args, scripts='main.py', **kwargs):
        args.fold = args.fold.split(',')
        args.fold = [int(i) for i in args.fold]

        args.device = args.device.split(',')
        args.device = [int(i) for i in args.device]

        fold = args.fold.copy()
        device = args.device.copy()
        print(f"FOLD: {fold} --- DEVICE: {device}")
        for k, v in kwargs.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                raise ValueError(f'No such attribute {k} in args')

        self.commands = []
        del args.exp_path
        del args.logs
        for f, d in zip(fold, device):
            args.fold = f
            args.device = d
            self.commands.append(
                ' '.join([f'python {scripts}'] + [f'--{k}' if isinstance(v, bool) else f'--{k} {v}' for k, v in
                                                  vars(args).items() if not isinstance(v, (bool, NoneType)) or
                                                  (isinstance(v, bool) and v)])
            )

    def run(self):
        # Create progress bar
        progress_bar = tqdm(total=len(self.commands), desc='Running Processes')
        processes = []

        def run_command(command):
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            processes.append(process)
            print(f'Starting Process {process.pid}，Executing command：{command}')
            out, err = process.communicate()
            if process.returncode == 0:
                print(f'Process {process.pid} execute successfully')
            else:
                print(f'Process {process.pid} execute failed')
                print(f'Process {process.pid} Standard error output：')
                print(err.decode())
                print(f'Process {process.pid} Executed command：{command}')
            progress_bar.update(1)

        # Signal handler for SIGINT (Ctrl+C)
        def signal_handler(sig, frame):
            print("\nReceived Ctrl+C, terminating all child processes...")
            for process in processes:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Send SIGTERM to the process group
                    print(f'Terminated Process {process.pid}')
                except ProcessLookupError:
                    # Process might have already terminated
                    continue
            progress_bar.close()
            exit(0)

        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)
        # Use ThreadPoolExecutor to manage subprocesses.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.commands)) as executor:
            futures = {executor.submit(run_command, command) for command in self.commands}

        # Close progress bar.
        progress_bar.close()
