# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from typing import Any, Dict, List, Tuple, Union
import os
from PIL import Image, ImageDraw

import torch
import numpy as np
import yaml

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L
# X = lambda i: int(gtsam.symbol(ord('x'), i))
# L = lambda i: int(gtsam.symbol(ord('l'), i))

from quadrics_multiview import groundtruth_quadrics
from utils import align_times, read_trajectory, align_trajectory




class Instances(object):
    """
    Implement class to handle multiple instances.
    """
    def __init__(self, instances: List["Instance"], calibration) -> None:
        self.instances = instances
        self.calibration = calibration
        self.cam_ids = self._get_cam_ids()
        self.bbox_ids = self._get_box_ids()
        self.values = None

    @classmethod
    def load_dataset(cls, path):
        """
        Factory method to load Instaces
        given the directory path.
        """
        # Load Intrinsics
        intrinsics_file = open(path+"calibration.yaml", 'r')
        intrinsics  = yaml.load(intrinsics_file)
        intrinsics = gtsam.Cal3_S2(intrinsics["fx"], intrinsics["fy"],
                                            0.0, intrinsics["cx"], intrinsics["cy"])

        orb_keys, orb_poses = read_trajectory(path+"CameraTrajectory_ORBVO.txt")
        gt_keys, gt_poses = read_trajectory(path+"groundtruth.txt")

        matches = align_times(orb_keys, gt_keys) # align orb and gt timestamsp

        orb_keys, gt_keys = zip(*matches)
        orb_poses = [orb_poses[key] for key in orb_keys]
        gt_poses = [gt_poses[key] for key in gt_keys]

        gt_poses = align_trajectory(orb_poses, gt_poses) # align gt trajectory wrt to orb trajectory

        data = torch.load(path+"tum.pth", map_location=torch.device('cpu'))
        file_keys = data['predicted_boxes'].keys()
        file_keys = list(map(lambda x: x.replace('.png', ''), file_keys))

        instances_list = []

        for i in range(len(orb_keys)):
            orb_key = orb_keys[i]
            if orb_key in file_keys:
                key = orb_key+".png"
                instance = Instance(bbox = data['predicted_boxes'][key].numpy(),
                            image_key = int(data['image_keys'][key].numpy()[0]),
                            bbox_covar = data['predicted_covar_mats'][key].numpy(),
                            pose = orb_poses[i],
                            gt_pose = gt_poses[i],
                            object_key = [int(x) for x in data['predicted_instance_key'][key].numpy()],
                            image_path = path+'rgb/'+key)

                instances_list.append(instance)

        instances_list.sort(key = lambda x: int(x.image_key))

        return cls(instances_list, intrinsics)

    def __getattr__(self, name: str) -> List["Instance"]:
        return [instance.get(name) for instance in self.instances]

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instance":
        """
        Args:
            item: an index-like object and will be used to index instances.

        Returns:
            If `item` is a int, return the data in the corresponding field.
            Otherwise, returns current `Instances` where all instances are indexed by `item`.
        """
        if type(item)==slice:
            ret = Instances(self.instances[item], self.calibration)
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

    def get_gt(self) -> gtsam.Values:
        gt = gtsam.Values()
        for key, pose in zip(self.image_key, self.gt_pose):
            gt.insert(X(key), pose)

        return gt

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
        
    def get_bbox_std(self) -> dict:
        wh_quadric = {q: [] for q in self.bbox_ids}
        # Compute sum of width and height for each quadric using ALL bboxes
        for sample in self.instances:
            w_h_sum = (sample.bbox[:, 2:] - sample.bbox[:, :2]).sum(axis=1)
            for key, wh in zip(sample.object_key, w_h_sum):
                wh_quadric[key].append(wh)

        std_quadric = {q: gtsam.noiseModel.Diagonal.Sigmas(np.array([np.std(wh_quadric[q])] * 4, dtype=float))
                            for q in wh_quadric.keys()}

        return std_quadric
    
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

    def __init__(self, **kwargs: Any):  # This Any is an class import from typing
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

    def __setattr__(self, name: str, val: Any) -> None:  # Overwriting default python function
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


class Instances(object):
    """
    Implement class to handle multiple instances.
    """

    def __init__(self, instances: List[Instance], calibration) -> None:
        self.instances = instances
        self.calibration = calibration
        self.cam_ids = self._get_cam_ids()
        self.bbox_ids = self._get_box_ids()
        self.values = None

    @classmethod
    def load_dataset(cls, path):
        """
        Factory method to load Instaces
        given the directory path.
        """
        # Load Intrinsics
        intrinsics_file = open(os.path.join(path, '..', "calibration.yaml"), 'r')
        intrinsics = yaml.safe_load(intrinsics_file)
        intrinsics = gtsam.Cal3_S2(intrinsics["fx"], intrinsics["fy"], 0.0, intrinsics["cx"], intrinsics["cy"])

        orb_keys, orb_poses = read_trajectory(os.path.join(path, '..', "trajectories", "CameraTrajectory_ORBVO.txt"))
        gt_keys, gt_poses = read_trajectory(os.path.join(path, "groundtruth.txt"))

        matches = align_times(orb_keys, gt_keys)  # Align orb and gt timestamsp

        orb_keys, gt_keys = zip(*matches)
        orb_poses = [orb_poses[key] for key in orb_keys]
        gt_poses = [gt_poses[key] for key in gt_keys]

        # gt_poses = align_trajectory(orb_poses, gt_poses)  # align gt trajectory wrt to orb trajectory

        data = torch.load(os.path.join(path, '..', 'probabilistic_detections', "detr_en.pth"),
                          map_location=torch.device('cpu'))
        file_keys = data['predicted_boxes'].keys()
        file_keys = list(map(lambda x: x.replace('.png', ''), file_keys))

        instances_list = []

        for i in range(len(orb_keys)):
            orb_key = orb_keys[i]
            if orb_key in file_keys:
                key = orb_key + ".png"
                instance = Instance(bbox=data['predicted_boxes'][key].numpy(),
                                    image_key=int(data['image_keys'][key].numpy()[0]),
                                    bbox_covar=data['predicted_covar_mats'][key].numpy(),
                                    pose=orb_poses[i],
                                    gt_pose=gt_poses[i],
                                    object_key=[int(x) for x in data['predicted_instance_key'][key].numpy()],
                                    image_path=os.path.join(path, 'rgb', key))

                instances_list.append(instance)

        instances_list.sort(key=lambda x: int(x.image_key))

        return cls(instances_list, intrinsics)

    def __getattr__(self, name: str) -> List[Instance]:
        return [instance.get(name) for instance in self.instances]

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> Instance:
        """
        Args:
            item: an index-like object and will be used to index instances.

        Returns:
            If `item` is a int, return the data in the corresponding field.
            Otherwise, returns current `Instances` where all instances are indexed by `item`.
        """
        if type(item) == slice:
            ret = Instances(self.instances[item], self.calibration)
            return ret

        return self.instances[item]

    def toValues(self) -> gtsam.Values:
        if self.values == None:
            self.values = gtsam.Values()

            for key, pose in zip(self.image_key, self.pose):
                self.values.insert(X(key), pose)

            groundtruth_quadrics(self.instances, self.values, self.calibration)

            return self.values

        else:
            return self.values

    def get_gt(self) -> gtsam.Values:
        gt = gtsam.Values()
        for key, pose in zip(self.image_key, self.gt_pose):
            gt.insert(X(key), pose)

        return gt

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

    def get_bbox_std(self) -> dict:
        wh_quadric = {q: [] for q in self.bbox_ids}
        # Compute sum of width and height for each quadric using ALL bboxes
        for sample in self.instances:
            w_h_sum = (sample.bbox[:, 2:] - sample.bbox[:, :2]).sum(axis=1)
            for key, wh in zip(sample.object_key, w_h_sum):
                wh_quadric[key].append(wh)

        std_quadric = {q: gtsam.noiseModel.Diagonal.Sigmas(np.array([np.std(wh_quadric[q])] * 4, dtype=float))
                       for q in wh_quadric.keys()}

        return std_quadric

    def __len__(self) -> int:
        return len(self.instances)
