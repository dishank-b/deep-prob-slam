# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from typing import Any, Dict, List, Tuple, Union
from PIL import Image, ImageDraw

import torch
import numpy as np

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L

from quadrics_multiview import groundtruth_quadrics

class Instances:
    """
    Implement class to handle multiple isntances. 
    """
    def __init__(self, instances: List["Instance"], calibration: dict = None) -> None:
        self.instances = instances
        if calibration!=None:
            self.calibration = gtsam.Cal3_S2(calibration["fx"], calibration["fy"],
                                            0.0, calibration["cx"], calibration["cy"])
        else:
            self.calibration = calibration
        self.values = None
        self.cam_ids = self._get_cam_ids()
        self.bbox_ids = self._get_box_ids()
    
    def __getattr__(self, name: str) -> List["Instance"]:
        return [instance.get(name) for instance in self.instances]

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instance":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item)==slice:
            ret = Instances(self.instances[item])
            ret.calibration = self.calibration
            return ret

        return self.instances[item]

    def toValues(self) -> gtsam.Values:
        if self.values==None:
            self.values = gtsam.Values()

            for key, pose in zip(self.image_key, self.pose):
                self.values.insert(X(key), pose)

            groundtruth_quadrics(self.instances, self.values, self.calibration)

            return self.values
        
        else:
            return self.values

    def _get_cam_ids(self):
        cam_ids = [instance.image_key for instance in self.instances]
        return cam_ids

    def _get_box_ids(self):
        bbox_ids = []
        for instance in self.instances:
            for obj_id in instance.object_key:
                if obj_id not in bbox_ids:
                    bbox_ids.append(int(obj_id))
    
        return bbox_ids

    def __len__(self) -> int:
        return len(self.instances)

    


class Instance:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same `__len__` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/Get a field:
       instances.gt_boxes = Boxes(...)
       print(instances.pred_masks)
       print('gt_masks' in instances)
    2. `len(instances)` returns the number of instances
    3. Indexing: `instances[indices]` will apply the indexing on all the fields
       and returns a new `Instances`.
       Typically, `indices` is a binary vector of length num_instances,
       or a vector of integer indices.
    """

    def __init__(self, **kwargs: Any): #This Any is an class import from typing
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        # self._fields: Dict[str, Any] = {}
        self._fields = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None: #Overwriting default python function
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        # data_len = len(value)
        # if len(self._fields):
        #     assert (
        #         len(self) == data_len
        #     ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Converts list fields to tensor
    def toTensor(self):
        for k, v in self._fields.items():
            if isinstance(v, list):
                self._fields[k] = torch.tensor(v)

    def toList(self):
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                self._fields[k] = v.cpu().numpy()
            if isinstance(v, Boxes):
                self._fields[k] = v.tensor.cpu().numpy()


    # Tensor-like methods
    def to(self, device: str) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instance(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instance":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        ret = Instance()
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError("Empty Instances does not support __len__!")

    @staticmethod
    def cat(instance_lists: List["Instance"]) -> "Instance":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instance) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        # s += "image_height={}, ".format(self._image_size[0])
        # s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, v)
        s += "])"
        return s