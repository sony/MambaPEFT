from typing import List, Optional, Union

from mmpretrain.registry import DATASETS
from mmpretrain.datasets.custom import CustomDataset
import os

@DATASETS.register_module()
class Vtab1K(CustomDataset):
    """
    Args:
        sub_dataset_name (str): Each folder name in the Vtab-1K, e.g. cifar100.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        split (str): The dataset split, supports "train", "val" and "test".
            Default to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """  # noqa: E501

    def __init__(self,
                 sub_dataset_name,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):

        super().__init__(
            data_root=os.path.join(data_root,sub_dataset_name),
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
