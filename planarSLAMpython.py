import gtsam
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import numpy as np
from scipy.spatial.distance import mahalanobis
import utils


def make_graph(file_name, odometry_noise):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    landmark_keys = []

    with open(file_name) as f:
        rows = f.readlines()
        for row in rows:
            row = row.strip().split(" ")
            row = [float(i) if idx != 0 else i for idx, i in enumerate(row)]      
            if row[0] == "VERTEX2":
                initial_estimate.insert(int(row[1]), gtsam.Pose2(row[2], row[3], row[4]))
            if row[0] == "EDGE2":
                # keys["Xs"].append(X(row[1]))
                # keys["Xs"].append(X(row[2]))
                odometry = gtsam.Pose2(row[3], row[4], row[5])
                graph.add(gtsam.BetweenFactorPose2(int(row[1]), int(row[2]), odometry, odometry_noise))
            if row[0] == "BR": 
                MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([row[5], row[6]]))
                graph.add(gtsam.BearingRangeFactor2D(int(row[1]), L(int(row[2])), gtsam.Rot2(row[3]), row[4], MEASUREMENT_NOISE))
                if not initial_estimate.exists(L(int(row[2]))):
                    initial_estimate.insert(L(int(row[2])), initial_estimate.atPose2(int(row[1])).transformFrom(gtsam.Point2(np.cos(row[3])*row[4], np.sin(row[3])*row[4])))
                    landmark_keys.append(L(int(row[2])))

    return graph, initial_estimate, landmark_keys


odometryNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 2*np.pi/180]))
graph, initial, Ls = make_graph("example.txt", odometryNoise)

priorMean = initial.atPose2(0)
priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 2*np.pi/180]))
graph.add(gtsam.PriorFactorPose2(0, priorMean, priorNoise))

utils.plot_trajectory(0, initial)
# utils.plot_landmarks(0, initial, Ls, marginals)

params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)


result = optimizer.optimize()
marginals = gtsam.Marginals(graph, result)

utils.plot_trajectory(0, result, marginals=marginals)
utils.plot_landmarks(0, result, Ls, marginals)

plt.axis('equal')
plt.show()