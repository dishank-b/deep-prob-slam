import numpy as np

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L

def init_quad_single_frame(bbox, camera_pose, camera_intrinsics):
    """
    Calculate q given one boudning box and camera pose. 
    Solves for Aq = 0. Where A formed using one bouding box from the the image. 
    Rank of A (4) is less than number of variables in q (10) to solve properly/
    Hence underdeterminent system. 
    """
    planes = []
    
    for i in range(bbox.lines().size()):
        planes.append(gtquadric.QuadricCamera.transformToImage(camera_pose, camera_intrinsics).transpose()@bbox.lines().at(i))
    A = []
    
    for plane in planes:
        a = plane[..., None]*np.ones((len(plane), len(plane)))*plane
        A.append(a)
    
    for i in range(len(A)):
        a = A[i]
        a = np.triu(2*a)-np.diag(np.diag(a))
        A[i] = a[np.triu_indices(len(a))]
    
    A = np.array(A)
    
    return A

def init_quad_multi_frames(bboxes, camera_poses, camera_intrinsics):
    """
    Calculate quadric q given boudning box measurement over multiple frames and
    respective camera poses. 
    Solves for Aq = 0. Where A formed using one bouding box from the the image. 
    Rank of A (4) is less than number of variables in q (10) to solve properly/
    Hence underdeterminent system. 

    Refer to Eq. 9 in the paper
    """
    A = []
    
    for box, camera_pose in zip(bboxes, camera_poses):
        A.append(init_quad_single_frame(gtquadric.AlignedBox2(*box), camera_pose, camera_intrinsics))

    A = np.concatenate(A)
    
    _, _, VT = np.linalg.svd(A)
    q = VT.T[:,-1]
    Q = np.zeros((4,4))
    Q[np.triu_indices(4)] = q
    Q = Q+Q.T-np.diag(np.diag(Q))
    
    return gtquadric.ConstrainedDualQuadric(Q)

def groundtruth_quadrics(instances, values: gtsam.Values, calibration):
    """
    Initialize the quadrics
    """
    obj_dict = {}
    
    for instance in instances:
        for obj_id, box in zip(instance.object_key, instance.bbox):
            if obj_id not in obj_dict:
                obj_dict[obj_id] = [[box, instance.pose]]
            else:
                obj_dict[obj_id].append([box, instance.pose])
    
    for obj_id, box_cam_pair in obj_dict.items():
        quadric = init_quad_multi_frames(*zip(*box_cam_pair), calibration)
        quadric.addToValues(values, L(obj_id))