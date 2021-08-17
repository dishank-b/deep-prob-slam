import gtsam
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import numpy as np
from scipy.spatial.distance import mahalanobis
import utils

np.random.seed(10) 

#---------Generating ground truth------------#

TIMESTEPS = 100
NUM_LANDMARKS = 50
DIM_FIELD = 100 # in meters
SENSOR_RANGE = 20.0 # in meters

# Generating robot GT trajectory

INITIAL_STATE = np.array([0.0, 0.0, 0.0])

trajectory = [INITIAL_STATE]

state = INITIAL_STATE
for t in range(TIMESTEPS):
    velo = np.random.normal(1.0, 0.4, 2)
    state = state + np.append(velo, 0.0)
    trajectory.append(state)

trajectory = np.array(trajectory)

plt.plot(*zip(*trajectory[:,:-1]), marker = "o")


# Generating landmarks
landmarks = []

for x in range(0, NUM_LANDMARKS*2, 10):
    for y in range(0,NUM_LANDMARKS*2, 10):
        landmarks.append((np.random.uniform(x, x+10), np.random.uniform(y, y+10)))

mask = []
for land_pose in landmarks:
    in_range = False
    for rob_pose in trajectory:
        if np.linalg.norm(rob_pose[:-1]-land_pose) < SENSOR_RANGE:
            in_range = True
    mask.append(in_range)

landmarks = np.array(landmarks)[mask]

plt.scatter(*zip(*landmarks), color="red")

# plt.show()
 

#--------- Solving SLAM --------- #
 
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05]))

graph = gtsam.NonlinearFactorGraph()

Xs = [X(i+1) for i in range(len(trajectory))]
Ls = [L(i+1) for i in range(len(landmarks))]

# factored_landmarks = []

# Adding odometry factors and landmark measurement factors
for idx, robo_pose in enumerate(trajectory):
    if idx==0:
        graph.add(gtsam.PriorFactorPose2(Xs[idx], gtsam.Pose2(*robo_pose), PRIOR_NOISE))

    if not (idx+1 == len(trajectory)):
        # odometry = trajectory[idx+1] - robo_pose
        # odometry[2] = np.arctan2(*trajectory[idx+1][:-1][::-1])-np.arctan2(*robo_pose[:-1][::-1])
        # odometry = gtsam.Pose2(*odometry)
        diff = trajectory[idx+1] - robo_pose
        d = np.linalg.norm(diff[:-1])
        delta = np.arctan2(diff[1],diff[0]) - robo_pose[2]
        odometry = gtsam.Pose2(d*np.cos(delta), d*np.sin(delta), diff[2]) # his odometry has to be wrt current robot frame, so directly taking different between two robot poses will give odom in ground frame. 

        graph.add(gtsam.BetweenFactorPose2(Xs[idx], Xs[idx+1], odometry, ODOMETRY_NOISE))

    for j, land_pose in enumerate(landmarks):
        dist = np.linalg.norm(robo_pose[:-1]-land_pose)
        if dist < SENSOR_RANGE:
            graph.add(gtsam.BearingRangeFactor2D(Xs[idx], Ls[j], gtsam.Rot2.fromAngle(np.arctan2(*land_pose[::-1])-np.arctan2(*robo_pose[:-1][::-1])), dist, MEASUREMENT_NOISE))
            # if j not in factored_landmarks:
            #     factored_landmarks.append(j)

# # Print graph
# print("Factor Graph:\n{}".format(graph))


# Create (deliberately inaccurate) initial estimate
initial_estimate = gtsam.Values()

for pose_id, pose in zip(Xs, trajectory):
    initial_estimate.insert(pose_id, gtsam.Pose2(*pose))

for land_idx, pose in zip(Ls, landmarks):
    initial_estimate.insert(land_idx, gtsam.Point2(*pose))

utils.plot_trajectory(0, initial_estimate)
utils.plot_landmarks(0, initial_estimate, Ls)   

# plt.axis('equal')
# plt.show()

# exit()


params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()
# print("\nFinal Result:\n{}".format(result))

# Calculate and print marginal covariances for all variables
marginals = gtsam.Marginals(graph, result)

utils.plot_trajectory(0, result, marginals=marginals)
utils.plot_landmarks(0, result, Ls, marginals)

esti_trajectory = np.array([result.atPose2(key).translation() for key in Xs])
esti_trajectory_covar = np.array([marginals.marginalCovariance(key) for key in Xs])
esti_landmarks = np.array([result.atPoint2(key) for key in Ls])
esti_landmarks_covar = np.array([marginals.marginalCovariance(key) for key in Ls])

print(esti_landmarks.shape, esti_landmarks_covar.shape, esti_trajectory.shape, esti_trajectory_covar.shape)

#------------ Evaluation ------------ #

mse_robo = np.array([np.linalg.norm(gt_robo_pose-esti_robo_pose) for gt_robo_pose, esti_robo_pose in zip(trajectory[:, :-1], esti_trajectory)]).mean() #ignoring orientation for MSE
mahalanobis_robo = np.array([mahalanobis(gt_robo_pose, esti_robo_pose, np.linalg.inv(covar)) for gt_robo_pose, esti_robo_pose, covar in zip(trajectory[:, :-1], esti_trajectory, esti_trajectory_covar[:, 0:2, 0:2])]).mean()

mse_landmarks = np.array([np.linalg.norm(gt_land_pose-esti_land_pose) for gt_land_pose, esti_land_pose in zip(landmarks, esti_landmarks)]).mean() #ignoring orientation for MSE
mahalanobis_landmark = np.array([mahalanobis(gt_land_pose, esti_land_pose, np.linalg.inv(covar)) for gt_land_pose, esti_land_pose, covar in zip(landmarks, esti_landmarks, esti_landmarks_covar)]).mean()

print("Robot Pose - MSE: {}, Mean Mahalanobis: {}".format(mse_robo, mahalanobis_robo))
print("Landmark Pose - MSE: {}, Mean Mahalanobis: {}".format(mse_landmarks, mahalanobis_landmark))

# print("Robot pose: ", np.array([[d,c] for d,c in zip(trajectory[:,:-1], esti_trajectory)]))
# print("Landmarks: ", np.array([[d,c] for d,c in zip(landmarks, esti_landmarks)]))

plt.axis('equal')
plt.show()
