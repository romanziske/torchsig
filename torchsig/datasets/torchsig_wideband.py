"""TorchSig Wideband Dataset
"""

from torchsig.transforms.target_transforms import ListTupleToDesc
from torchsig.transforms.transforms import Identity
from torchsig.utils.types import Signal, create_signal_data
from torchsig.datasets import conf
from torchsig.datasets.signal_classes import torchsig_signals
from typing import Callable, List, Optional
from pathlib import Path
import numpy as np
import pickle
import lmdb
import os


class TorchSigWideband:

    """The Official TorchSigWideband dataset

    Args:
        root (string): Root directory of dataset. A folder will be created for the requested version
            of the dataset, an mdb file inside contains the data and labels.
        train (bool, optional): If True, constructs the corresponding training set,
            otherwise constructs the corresponding val set
        impaired (bool, optional): If True, will construct the impaired version of the dataset,
            with data passed through a seeded channel model
        transform (callable, optional): A function/transform that takes in a complex64 ndarray
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target class (int) and returns a transformed version

    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        impaired: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        class_list: Optional[List] = None
    ):
        self.root = Path(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.train = train
        self.impaired = impaired
        self.class_list = torchsig_signals.class_list if class_list is None else class_list

        self.T = transform if transform else Identity()
        self.TT = target_transform if target_transform else Identity()

        cfg = ("Wideband" + ("Impaired" if impaired else "Clean") +
               ("Train" if train else "Val") + "Config")
        cfg = getattr(conf, cfg)()

        self.path = self.root / cfg.name  # type: ignore
       # Initialize LMDB
        self._init_lmdb()

    def __len__(self) -> int:
        return self.length

    def _get_data_label(self, idx: int):
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db, write=False) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx))

        with self.env.begin(db=self.label_db, write=False) as label_txn:
            label = pickle.loads(label_txn.get(encoded_idx))

        return iq_data, label

    def __getitem__(self, idx: int) -> tuple:
        iq_data, label = self._get_data_label(idx)

        signal = Signal(data=create_signal_data(
            samples=iq_data), metadata=(label))
        transformed = self.T(signal)

        # Handle multiview case
        if isinstance(transformed, list):
            samples = []
            for view in transformed:
                target = self.TT(view["metadata"])
                samples.append(view["data"]["samples"])

            return samples, target

        # Single view case
        target = self.TT(transformed["metadata"])
        return transformed["data"]["samples"], target

    def _init_lmdb(self):
        """Initialize LMDB environment - called in __init__ and after unpickling"""
        self.env = lmdb.open(str(self.path),
                             map_size=int(1e12),
                             max_dbs=2,
                             readonly=True,
                             lock=False,
                             readahead=False)
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")

        with self.env.begin(db=self.data_db, write=False) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __getstate__(self):
        """Return picklable state"""
        state = self.__dict__.copy()
        # Remove LMDB environment
        del state['env']
        del state['data_db']
        del state['label_db']
        return state

    def __setstate__(self, state):
        """Restore from pickle"""
        self.__dict__.update(state)
        # Reopen LMDB in new process
        self._init_lmdb()
        self.label_db = self.env.open_db(b"label")
