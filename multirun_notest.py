from utils.multi_runner import MultiRunner
from config import parse_args

runner = MultiRunner(args=parse_args(add_config=False), scripts='main_binary.py', no_test=True)
runner.run()