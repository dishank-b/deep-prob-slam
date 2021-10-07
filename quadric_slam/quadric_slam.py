import numpy as np
import matplotlib.pyplot as plt
import yaml
import glob

import gtsam 
from gtsam.symbol_shorthand import X, L
import gtsam_quadrics as gtquadric
import visualization

from instances import Instances


def get_data(dir):
    file_names = glob.glob(dir+"*.json")
    instance_list = []
    
    for file_name in file_names:
        file = open(file_name,'r')
        file = yaml.load(file)
        instance = Instances(**file)
        instance.pose = gtsam.Pose3(gtsam.Rot3.Quaternion(instance.pose[-1], *instance.pose[3:-1]), gtsam.Point3(*instance.pose[:3]))
        instance_list.append(instance)

    instance_list.sort(key = lambda x: int(x.image_key))
    
    return instance_list


def initialize_quadric(bbox, camera_pose, camera_intrinsics):
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
    _, _, VT = np.linalg.svd(A)
    q = VT.T[:,-1]
    Q = np.zeros((4,4))
    Q[np.triu_indices(4)] = q
    Q = Q+Q.T-np.diag(np.diag(Q))
    
    return gtquadric.ConstrainedDualQuadric(Q)

ODOM_SIGMA = 0.01
BOX_SIGMA = 3

instances = get_data("./data/preprocessed/")

#get data
file = open("./data/calibration.yaml", 'r')
intrinsics  = yaml.load(file)

# Xs = [X(i) for i in range(len(poses))]
# Ls = [L(i) for i in range(len(boundingboxes))]

#Make the graph
calibration = gtsam.Cal3_S2(intrinsics["fx"], intrinsics["fy"], 0.0, intrinsics["cx"], intrinsics["cy"])
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()


# define noise models
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1]*6, dtype=float))
odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([ODOM_SIGMA]*3 + [ODOM_SIGMA]*3, dtype=float))
bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([BOX_SIGMA]*4, dtype=float))

#Making the graph
for i, instance in enumerate(instances):
    if i==0:
        graph.add(gtsam.PriorFactorPose3(X(instance.image_key), instance.pose, prior_noise))
        initial_estimate.insert(X(instance.image_key), gtsam.Pose3(instance.pose))
    
    if i < len(instances)-1:
        relative_pose = instance.pose.between(instances[i+1].pose)
        # TODO: add noise to relative pose
        odometry_factor = gtsam.BetweenFactorPose3(X(instance.image_key), X(instances[i+1].image_key), relative_pose, odometry_noise)
        graph.add(odometry_factor)
        initial_estimate.insert(X(instances[i+1].image_key), initial_estimate.atPose3(X(instance.image_key)).compose(relative_pose))

    for obj_id, bbox in zip(instance.object_key, instance.bbox):
        box = gtquadric.AlignedBox2(*bbox)
        bbf = gtquadric.BoundingBoxFactor(box, calibration, X(instance.image_key), L(obj_id), bbox_noise)
        graph.add(bbf)
        if not initial_estimate.exists(L(obj_id)):
            quadric = initialize_quadric(box, initial_estimate.atPose3(X(instance.image_key)), calibration)
            # quadric = gtquadric.ConstrainedDualQuadric()
            quadric.addToValues(initial_estimate, L(obj_id))

# print(initial_estimate)
# exit()
# print(graph)

# define lm parameters
parameters = gtsam.LevenbergMarquardtParams()
parameters.setVerbosityLM("SUMMARY") # SILENT = 0, SUMMARY, TERMINATION, LAMBDA, TRYLAMBDA, TRYCONFIG, DAMPED, TRYDELTA : VALUES, ERROR 
parameters.setMaxIterations(100)
parameters.setlambdaInitial(1e-5)
parameters.setlambdaUpperBound(1e30)
parameters.setlambdaLowerBound(1e-8)
parameters.setRelativeErrorTol(1e-5)
parameters.setAbsoluteErrorTol(1e-5)

# create optimizer
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)

# run optimizer
result = optimizer.optimize()

gt_list = [instance.pose.translation() for instance in instances]
esti_list = [result.atPose3(X(instance.image_key)).translation() for instance in instances]

print("RMSE: ", np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(gt_list, esti_list)]).mean())





