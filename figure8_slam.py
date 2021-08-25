import gtsam
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import numpy as np
from scipy.spatial.distance import mahalanobis
from wandb.sdk.lib import disabled
import utils
import wandb as wb
import warnings
warnings.filterwarnings("ignore")
wb.login()

wb.init(
    # mode="disabled",
    project="deep-prob-slam",
    config = {
    "seed": 6,
    "SENSOR_RANGE" : 5.0, # in meters
    "dt" : 1,
    "VEL" : 1.0,
    "VEL_NOISE" : 0.15,
    "OMEGA" : 0.05,
    "OMEGA_NOISE" : 0.5,
    "USE_NOISE" : False,
    "TIMESTEPS" : int(2*2*np.pi/0.05),  # 2*2\pi/omega
    "INITIAL_STATE" : [0.0, 0.0, 0.0],
    "SPACING" : 3,
    "PRIOR_NOISE" : [0.05, 0.05, 2*np.pi/180],
    "ODOMETRY_NOISE" : [0.1, 0.1, 2*np.pi/180.0],
    "MEASUREMENT_NOISE" : [2*np.pi/180.0, 0.1], # bearing (angle) noise, distance noise.
    "uncalib_ODOMETRY_NOISE" : [0.2, 0.2, 4*np.pi/180.0],
    "uncalib_MEASUREMENT_NOISE" : [4*np.pi/180.0, 0.2], # bearing (angle) noise, distance noise.
    }
)

config = wb.config
np.random.seed(config.seed)

#---------Generating ground truth------------#

# Generating robot GT trajectory
trajectory = [config.INITIAL_STATE]
gt_odometry = []

state = np.array(config.INITIAL_STATE)
v = config.VEL
for t in range(0, config.TIMESTEPS, config.dt):
    if t < 2*np.pi/config.OMEGA:
        omega = config.OMEGA
    else:
        omega = -1*config.OMEGA

    theta = state[2]
    d_state = np.array([v*np.cos(theta), v*np.sin(theta), omega])*config.dt
    gt_odometry.append(d_state)
    state = state + d_state
    trajectory.append(state)

trajectory = np.array(trajectory)

# print(trajectory)
# plt.plot(*zip(*trajectory[:,:-1]), marker = "o")

# Generating landmarks
dim_field_max = int(np.max(trajectory[:, :-1]))
dim_field_min = int(np.min(trajectory[:, :-1]))
landmarks = []

for x in range(dim_field_min, dim_field_max, config.SPACING):
    for y in range(dim_field_min, dim_field_max, config.SPACING):
        landmarks.append((np.random.uniform(x, x+config.SPACING), np.random.uniform(y, y+config.SPACING)))

mask = []
for land_pose in landmarks:
    in_range = False
    for rob_pose in trajectory:
        if np.linalg.norm(rob_pose[:-1]-land_pose) < config.SENSOR_RANGE:
            in_range = True
    mask.append(in_range)

landmarks = np.array(landmarks)[mask]

# plt.scatter(*zip(*landmarks), color="red")

# -------- GT Plotting ---------# 

Xs = [X(i+1) for i in range(len(trajectory))]
Ls = [L(i+1) for i in range(len(landmarks))]

gt = gtsam.Values()

# robust to inital state values, just not give 0. 
for pose_id, pose in zip(Xs, trajectory):
    gt.insert(pose_id, gtsam.Pose2(*pose))

for land_idx, pose in zip(Ls, landmarks):
    gt.insert(land_idx, pose)

# utils.plot_trajectory(0, gt, label="GT")
# utils.plot_landmarks(0, gt, Ls, label="GT")  

# plt.savefig("{}_gt.png".format(run_id))
# plt.clf()

#--------- Solving SLAM --------- #

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.PRIOR_NOISE))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.ODOMETRY_NOISE))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.MEASUREMENT_NOISE)) # bearing (angle) noise, distance noise.
uncalib_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.uncalib_ODOMETRY_NOISE))
uncalib_MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.uncalib_MEASUREMENT_NOISE)) # bearing (angle) noise, distance noise.

# Graph to be solves, where all the nodes will be added
graph = gtsam.NonlinearFactorGraph()

# inital guess of values for optimizer
# initial_estimate = gtsam.Values()

# Adding odometry factors and landmark measurement factors
for idx, robo_pose in enumerate(trajectory):
    if idx==0:
        graph.add(gtsam.PriorFactorPose2(Xs[idx], gtsam.Pose2(*robo_pose), PRIOR_NOISE))
        # initial_estimate.insert(Xs[idx], gtsam.Pose2(*robo_pose))

    if idx>0:
        diff = robo_pose - trajectory[idx-1]
        d = np.linalg.norm(diff[:-1]) 
        delta = np.arctan2(diff[1],diff[0]) - trajectory[idx-1][2]
        odometry = np.array([d*np.cos(delta), d*np.sin(delta), diff[2]])
        odometry = np.random.multivariate_normal(odometry, ODOMETRY_NOISE.covariance())
        odometry = gtsam.Pose2(*odometry) # his odometry has to be wrt current robot frame, so directly taking different between two robot poses will give odom in ground frame. 
        # graph.add(gtsam.BetweenFactorPose2(Xs[idx-1], Xs[idx], odometry, ODOMETRY_NOISE))
        graph.add(gtsam.BetweenFactorPose2(Xs[idx-1], Xs[idx], odometry, uncalib_ODOMETRY_NOISE))
        # initial_estimate.insert(Xs[idx], initial_estimate.atPose2(Xs[idx-1]).transformFrom(odometry_np))
 

    for j, land_pose in enumerate(landmarks):
        diff = land_pose - robo_pose[:-1]
        d = np.linalg.norm(diff)
        delta = np.arctan2(diff[1],diff[0]) - robo_pose[-1]
        # sampling
        d = np.random.normal(d, MEASUREMENT_NOISE.sigmas()[1])
        delta = np.random.normal(delta, MEASUREMENT_NOISE.sigmas()[0])
        if d < config.SENSOR_RANGE:
            # graph.add(gtsam.BearingRangeFactor2D(Xs[idx], Ls[j], gtsam.Rot2.fromAngle(delta), d, MEASUREMENT_NOISE))
            graph.add(gtsam.BearingRangeFactor2D(Xs[idx], Ls[j], gtsam.Rot2.fromAngle(delta), d, uncalib_MEASUREMENT_NOISE))
            # if not initial_estimate.exists(Ls[j]):
                # initial_estimate.insert(Ls[j], initial_estimate.atPose2(Xs[idx]).transformFrom(gtsam.Point2(d*np.cos(delta), d*np.sin(delta))))

initial_estimate = gtsam.Values()
for pose_id, pose in zip(Xs, trajectory):
    # initial_estimate.insert(pose_id, gtsam.Pose2(*pose))
    initial_estimate.insert(pose_id, gtsam.Pose2(np.array(config.INITIAL_STATE)+0.05))

for land_idx, pose in zip(Ls, landmarks):
    # initial_estimate.insert(land_idx, pose)
    initial_estimate.insert(land_idx, gtsam.Point2(np.array(config.INITIAL_STATE[:-1])+0.05))

params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()

# Calculate and print marginal covariances for all variables
marginals = gtsam.Marginals(graph, result)

utils.plot_trajectory(1, gt, label="GT")
utils.plot_trajectory(1, result, marginals=marginals, label="Estimated")
utils.plot_landmarks(1, result, Ls, marginals, label="Estimated")

# plt.savefig("{}_estimated.png".format(run_id))

esti_trajectory = np.array([result.atPose2(key).translation() for key in Xs])
esti_trajectory_covar = np.array([marginals.marginalCovariance(key) for key in Xs])
esti_landmarks = np.array([result.atPoint2(key) for key in Ls])
esti_landmarks_covar = np.array([marginals.marginalCovariance(key) for key in Ls])

#------------ Evaluation ------------ #

rmse_robo = np.array([np.linalg.norm(gt_robo_pose-esti_robo_pose) for gt_robo_pose, esti_robo_pose in zip(trajectory[:, :-1], esti_trajectory)]).mean() #ignoring orientation for MSE
mahalanobis_robo = np.array([mahalanobis(gt_robo_pose, esti_robo_pose, np.linalg.inv(covar)) for gt_robo_pose, esti_robo_pose, covar in zip(trajectory[:, :-1], esti_trajectory, esti_trajectory_covar[:, 0:2, 0:2])]).mean()

rmse_landmarks = np.array([np.linalg.norm(gt_land_pose-esti_land_pose) for gt_land_pose, esti_land_pose in zip(landmarks, esti_landmarks)]).mean() #ignoring orientation for MSE
mahalanobis_landmark = np.array([mahalanobis(gt_land_pose, esti_land_pose, np.linalg.inv(covar)) for gt_land_pose, esti_land_pose, covar in zip(landmarks, esti_landmarks, esti_landmarks_covar)]).mean()

print("Robot Pose - RMSE: {}, Mean Mahalanobis: {}".format(rmse_robo, mahalanobis_robo))
print("Landmark Pose - RMSE: {}, Mean Mahalanobis: {}".format(rmse_landmarks, mahalanobis_landmark))

wb.log({"Rob Pose RMSE": rmse_robo, "Robo Mahalanobis": mahalanobis_robo , "Landmark Pose RMSE":rmse_landmarks, "Landmark Mahalanobis":mahalanobis_landmark})
wb.log({"Trajectory": wb.Image(plt)})

wb.finish()

plt.show()
