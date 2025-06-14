import logging
import warnings
from collections.abc import Callable

import torch

from monai.config import IgniteInfo
from monai.data import CSVSaver, decollate_batch
from monai.utils import ImageMetaKey as Key
from monai.utils import evenly_divisible_all_gather, min_version, optional_import, string_list_all_gather
from .logger import CSVLogs

idist, _ = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")
Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")


class ClassificationSaver:
    """
    Event handler triggered on completing every iteration to save the classification predictions as CSV file.
    If running in distributed data parallel, only saves CSV file in the specified rank.

    """

    def __init__(
            self,
            output_dir: str = "./",
            filename: str = "predictions.csv",
            overwrite: bool = True,
            batch_transform: Callable = lambda x: x,
            output_transform: Callable = lambda x: x,
            name: str | None = None,
            save_rank: int = 0,
            saver: CSVLogs | None = None,
    ) -> None:
        """
        Args:
            output_dir: if `saver=None`, output CSV file directory.
            filename: if `saver=None`, name of the saved CSV file name.
            overwrite: if `saver=None`, whether to overwriting existing file content, if True,
                will clear the file before saving. otherwise, will append new content to the file.
            batch_transform: a callable that is used to extract the `meta_data` dictionary of
                the input images from `ignite.engine.state.batch`. the purpose is to get the input
                filenames from the `meta_data` and store with classification results together.
                `engine.state` and `batch_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            output_transform: a callable that is used to extract the model prediction data from
                `ignite.engine.state.output`. the first dimension of its output will be treated as
                the batch dimension. each item in the batch will be saved individually.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.
            save_rank: only the handler on specified rank will save to CSV file in multi-gpus validation,
                default to 0.
            saver: the saver instance to save classification results, if None, create a CSVSaver internally.
                the saver must provide `save_batch(batch_data, meta_data)` and `finalize()` APIs.

        """
        self.save_rank = save_rank
        self.output_dir = output_dir
        self.filename = filename
        self.overwrite = overwrite
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.saver = saver

        self.logger = logging.getLogger(name)
        self._name = name
        self._outputs = []

    def attach(self, engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self._started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self._started)
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        if not engine.has_event_handler(self._finalize, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self._finalize)

    def _started(self, _engine) -> None:
        """
        Initialize internal buffers.

        Args:
            _engine: Ignite Engine, unused argument.

        """
        self._outputs = []

    def __call__(self, engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        meta_data = self.batch_transform(engine.state.batch)
        if isinstance(meta_data, dict):
            # decollate the `dictionary of list` to `list of dictionaries`
            meta_data = decollate_batch(meta_data)
        engine_output = self.output_transform(engine.state.output)
        epoch = engine.state.epoch
        for m, *o in zip(meta_data, *engine_output):
            for i in range(len(o)):
                if isinstance(o[i], torch.Tensor):
                    o[i] = o[i].detach().cpu()
            self._outputs.append({
                "epoch": epoch,
                "filename": m.get(Key.FILENAME_OR_OBJ),
                "output": o[0].numpy(),
                "pred": o[0].argmax().item(),
                "label": o[1],
            })

    def _finalize(self, _engine) -> None:
        """
        All gather classification results from ranks and save to CSV file.

        Args:
            _engine: Ignite Engine, unused argument.
        """
        ws = idist.get_world_size()
        if self.save_rank >= ws:
            raise ValueError("target save rank is greater than the distributed group size.")

        outputs = self._outputs
        if ws > 1:
            outputs = string_list_all_gather(outputs)

        # save to CSV file only in the expected rank
        if idist.get_rank() == self.save_rank:
            saver = self.saver or CSVLogs(
                dir_path=self.output_dir, file_name=self.filename, overwrite=self.overwrite,
            )
            saver.cache(outputs)
            saver.write(save_key="epoch")
