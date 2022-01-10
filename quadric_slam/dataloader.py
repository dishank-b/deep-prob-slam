import numpy as np
import glob
import yaml
import torch
import gtsam
from instances import Instances, Instance


def tum_raw(dir, intrinsics):
    """
    Read the data provided. 
    Instance - Class to handle each frame and it's associated attributes
    instance.pose - Camera pose wrt world coordinates
    instance.image_key - id for the frame
    instance.object_key - [K] ids of N object detected in the frame
    instance.bbox - [K,4] bouding boxes of N object detected in the frame
    instance.path - path of rgb image in the data directory

    Returns:
    instance_list - [Instance] - N instances. 
    """
    file_names = glob.glob(dir+"*.json")
    instance_list = []
    
    for file_name in file_names:
        file = open(file_name,'r')
        file = yaml.load(file)
        instance = Instance(**file)
        instance.pose = gtsam.Pose3(gtsam.Rot3.Quaternion(instance.pose[-1], *instance.pose[3:-1]), gtsam.Point3(*instance.pose[:3]))
        instance_list.append(instance)

    instance_list.sort(key = lambda x: int(x.image_key))
    
    return Instances(instance_list, intrinsics)



def tum_uncertainty(path, intrinsics):
    """
    Read the data with bounding box uncertainty estimates available. 
    """

    data = torch.load(path, map_location=torch.device('cpu'))

    instance_list = []

    for key in data['predicted_boxes'].keys():
        instance = Instance(bbox = data['predicted_boxes'][key].numpy(),
                    image_key = int(data['image_keys'][key].numpy()[0]),
                    bbox_covar = data['predicted_covar_mats'][key].numpy(),
                    pose = data['camera_pose'][key].numpy()[0],
                    object_key = [int(x) for x in data['predicted_instance_key'][key].numpy()],
                    image_path = 'rgbd_dataset_freiburg3_long_office_household/rgb/'+key)

        instance.pose = gtsam.Pose3(gtsam.Rot3.Quaternion(instance.pose[-1], *instance.pose[3:-1]), gtsam.Point3(*instance.pose[:3]))
        instance_list.append(instance)

    instance_list.sort(key = lambda x: int(x.image_key))
    
    return Instances(instance_list, intrinsics)


