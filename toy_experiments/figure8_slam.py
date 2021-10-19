import gtsam
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
from wandb.sdk.lib import disabled
import utils
import eval_metrices as metrices
import wandb as wb
import warnings
warnings.filterwarnings("ignore")
wb.login()


runs = 20 

for run_id in range(runs):

    print("Run Id: {}".format(run_id))

    #=============Config=============#

    wb.init(
        # mode="disabled",
        project="toy-prob-slam",
        entity="deeprobslam",
        group="calibrated",
        # group="calibrated",
        config = { 
        "run_id" : run_id,
        "GT_SEED": 6,
        "SENSOR_RANGE" : 5.0, # in meters
        "dt" : 1,
        "VEL" : 1.0,
        "VEL_NOISE" : 0.15,
        "OMEGA" : 0.05,
        "OMEGA_NOISE" : 0.5,
        "USE_NOISE" : False,
        "TIMESTEPS" : int(2*2*np.pi/0.05),  # 2*2\pi/omega
        "INITIAL_STATE" : [0.0, 0.0, 0.0],
        "SPACING" : 5,
        "PRIOR_NOISE" : [0.05, 0.05, 2*np.pi/180],
        "ODOMETRY_NOISE" : [0.1, 0.1, 2*np.pi/180.0],
        "MEASUREMENT_NOISE" : [2*np.pi/180.0, 0.1], # bearing (angle) noise, distance noise.
        "uncalib_ODOMETRY_NOISE" : [0.05, 0.05, np.pi/180.0],
        "uncalib_MEASUREMENT_NOISE" : [4*np.pi/180.0, 0.2], # bearing (angle) noise, distance noise.
        }
    )

    wb.run.name = str(run_id)

    config = wb.config

    #===============Generating ground truth===============#

    np.random.seed(config.GT_SEED)

    #---------Generating robot GT trajectory-------------#
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

    #----------Generating landmarks-------------------#
    dim_field_max = int(np.max(trajectory[:, :-1]))
    dim_field_min = int(np.min(trajectory[:, :-1]))
    landmarks = []

    # one landmark randomly places in square of size SPACING*SPACING
    for x in range(dim_field_min, dim_field_max, config.SPACING):
        for y in range(dim_field_min, dim_field_max, config.SPACING):
            landmarks.append((np.random.uniform(x, x+config.SPACING), np.random.uniform(y, y+config.SPACING)))

    # selecting only the landmarks which is in range of robot
    mask = []
    for land_pose in landmarks:
        in_range = False
        for rob_pose in trajectory:
            if np.linalg.norm(rob_pose[:-1]-land_pose) < config.SENSOR_RANGE:
                in_range = True
        mask.append(in_range)

    landmarks = np.array(landmarks)[mask]

    # plt.scatter(*zip(*landmarks), color="red")

    #=================================================================================#

    # ==============GT Plotting ==============# 

    Xs = [X(i+1) for i in range(len(trajectory))]
    Ls = [L(i+1) for i in range(len(landmarks))]

    gt = gtsam.Values()

    # robust to inital state values, just not give 0. 
    for pose_id, pose in zip(Xs, trajectory):
        gt.insert(pose_id, gtsam.Pose2(*pose))

    for land_idx, pose in zip(Ls, landmarks):
        gt.insert(land_idx, pose)


    utils.plot_trajectory(2, gt, label="GT")
    # utils.plot_landmarks(0, gt, Ls, label="GT")  

    # plt.savefig("{}_gt.png".format(run_id))
    # plt.clf()

    #======================Solving SLAM ======================#

    np.random.seed(run_id)

    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.PRIOR_NOISE))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.ODOMETRY_NOISE))
    MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.MEASUREMENT_NOISE)) # bearing (angle) noise, distance noise.
    uncalib_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.uncalib_ODOMETRY_NOISE))
    uncalib_MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.uncalib_MEASUREMENT_NOISE)) # bearing (angle) noise, distance noise.

    # Graph to be solves, where all the nodes will be added
    graph = gtsam.NonlinearFactorGraph()

    # inital guess of values for optimizer
    initial_estimate = gtsam.Values()

    #---------Adding odometry factors and landmark measurement factors----------------#
    for idx, robo_pose in enumerate(trajectory):
        if idx==0:
            graph.add(gtsam.PriorFactorPose2(Xs[idx], gtsam.Pose2(*robo_pose), PRIOR_NOISE))
            initial_estimate.insert(Xs[idx], gtsam.Pose2(*robo_pose))

        if idx>0:
            diff = robo_pose - trajectory[idx-1]
            d = np.linalg.norm(diff[:-1]) 
            delta = np.arctan2(diff[1],diff[0]) - trajectory[idx-1][2]
            odometry = np.array([d*np.cos(delta), d*np.sin(delta), diff[2]])
            odometry = np.random.multivariate_normal(odometry, ODOMETRY_NOISE.covariance())
            odometry = gtsam.Pose2(*odometry) # this odometry has to be wrt current robot frame, so directly taking different between two robot poses will give odom in ground frame. 
            graph.add(gtsam.BetweenFactorPose2(Xs[idx-1], Xs[idx], odometry, ODOMETRY_NOISE))
            # graph.add(gtsam.BetweenFactorPose2(Xs[idx-1], Xs[idx], odometry, uncalib_ODOMETRY_NOISE))
            initial_estimate.insert(Xs[idx], initial_estimate.atPose2(Xs[idx-1]).compose(odometry))
    

        for j, land_pose in enumerate(landmarks):
            diff = land_pose - robo_pose[:-1]
            d = np.linalg.norm(diff)
            delta = np.arctan2(diff[1],diff[0]) - robo_pose[-1]
            if d < config.SENSOR_RANGE:
                # sampling
                d = np.random.normal(d, MEASUREMENT_NOISE.sigmas()[1])
                delta = np.random.normal(delta, MEASUREMENT_NOISE.sigmas()[0])
                graph.add(gtsam.BearingRangeFactor2D(Xs[idx], Ls[j], gtsam.Rot2.fromAngle(delta), d, MEASUREMENT_NOISE))
                # graph.add(gtsam.BearingRangeFactor2D(Xs[idx], Ls[j], gtsam.Rot2.fromAngle(delta), d, uncalib_MEASUREMENT_NOISE))
                if not initial_estimate.exists(Ls[j]):
                    initial_estimate.insert(Ls[j], initial_estimate.atPose2(Xs[idx]).transformFrom(gtsam.Point2(d*np.cos(delta), d*np.sin(delta))))

    # utils.plot_trajectory(0, initial_estimate, label="initial")
    # utils.plot_landmarks(0, initial_estimate, Ls, label="initial")

    #------------ Optimization of Graph ---------------#
    # initial_estimate = gtsam.Values()
    # for pose_id, pose in zip(Xs, trajectory):
    #     # initial_estimate.insert(pose_id, gtsam.Pose2(*pose))
    #     initial_estimate.insert(pose_id, gtsam.Pose2(np.array(config.INITIAL_STATE)+0.05))

    # for land_idx, pose in zip(Ls, landmarks):
    #     # initial_estimate.insert(land_idx, pose)
    #     initial_estimate.insert(land_idx, gtsam.Point2(np.array(config.INITIAL_STATE[:-1])+0.05))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    result = optimizer.optimize() # solution of graph
    marginals = gtsam.Marginals(graph, result) # Calculate and print marginal covariances for all variables

    utils.plot_trajectory(2, result, marginals=marginals, label="Estimated")
    utils.plot_landmarks(2, result, Ls, marginals, label="Estimated")

    # plt.savefig("{}_estimated.png".format(run_id))


    # location of robot and landmarks with associated predicted covariance
    esti_trajectory = np.array([np.concatenate((result.atPose2(key).translation(), np.array([result.atPose2(key).theta()]))) for key in Xs])
    esti_trajectory_covar = np.array([marginals.marginalCovariance(key) for key in Xs])
    esti_landmarks = np.array([result.atPoint2(key) for key in Ls])
    esti_landmarks_covar = np.array([marginals.marginalCovariance(key) for key in Ls])

    # print(esti_trajectory, esti_trajectory_covar)

    #------------ Evaluation ------------ #

    rmse_robo = metrices.rmse(trajectory[:,:-1], esti_trajectory[:,:-1])
    mahalanobis_robo = metrices.mahalanobis(trajectory[:,:-1], esti_trajectory[:,:-1], esti_trajectory_covar[:, 0:2, 0:2])
    nll_robo = metrices.nll(trajectory[:,:-1], esti_trajectory[:,:-1], esti_trajectory_covar[:, 0:2, 0:2])
    ece_robo = metrices.ece(trajectory[:,:-1], esti_trajectory[:,:-1], esti_trajectory_covar[:, 0:2, 0:2], plot=True)
    ece_robo_full = metrices.ece(trajectory, esti_trajectory, esti_trajectory_covar)

    rmse_landmarks = metrices.rmse(landmarks, esti_landmarks)
    mahalanobis_landmark = metrices.mahalanobis(landmarks, esti_landmarks, esti_landmarks_covar)
    nll_landmark = metrices.nll(landmarks, esti_landmarks, esti_landmarks_covar)
    ece_landmark = metrices.ece(landmarks, esti_landmarks, esti_landmarks_covar)

    nll_odom = metrices.odom_nll(trajectory, esti_trajectory, uncalib_ODOMETRY_NOISE.covariance())

    print("Robot Pose - RMSE: {}, Mean Mahalanobis: {}, Robo ECE: {}, Robo NLL: {}".format(rmse_robo, mahalanobis_robo, ece_robo, nll_robo))
    print("Landmark Pose - RMSE: {}, Mean Mahalanobis: {}, ECE: {}, NLL: {} ".format(rmse_landmarks, mahalanobis_landmark, ece_landmark, nll_landmark))

    wb.log({"Robo Poses": esti_trajectory,
            "Landmark Poses" : esti_landmarks,
            "Robo Poses Covar": esti_trajectory_covar,
            "Landmark Poses Covar": esti_landmarks_covar,
            "GT Robo poses": trajectory,
            "GT Landmark poses": landmarks,
            "Robo Pose RMSE": rmse_robo, 
            "Robo Mahalanobis": mahalanobis_robo,
            "Robo NLL" : nll_robo, 
            "Robo ECE": ece_robo,
            "Robo ECE full": ece_robo_full,
            "Landmark Pose RMSE":rmse_landmarks, 
            "Landmark Mahalanobis":mahalanobis_landmark, 
            "Landmark NLL": nll_landmark,
            "Landmark ECE": ece_landmark,
            "Odom ECE": nll_odom,
            "Trajectory": wb.Image(plt)}
            )

    wb.finish()

    plt.clf()

    # plt.show()
