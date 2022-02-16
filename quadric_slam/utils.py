import numpy as np
from scipy.optimize import linear_sum_assignment
import gtsam

def align_times(first, second, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first -- first list of timestamps
    second -- second list of timestamps
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((first_stamp,second_stamp),(first_stamp,second_stamp))

    """

    firstM = np.tile(np.array(first, dtype='float'), (len(second), 1))
    secondM = np.tile(np.expand_dims(np.array(second, dtype='float'), 0).T, (1, len(first)))
    cost = np.abs(firstM - (secondM + offset))
    assignment = linear_sum_assignment(cost)
    valid = cost[assignment] < max_difference

    secondI = assignment[0][valid]
    firstI = assignment[1][valid]

    matches = [(first[a], second[b]) for a, b in zip(firstI, secondI)]
    matches.sort()
    return matches

def align_trajectory(traj1, traj2):
    """
    Align trajectory 2 wrt to trajectory 1
    """
    assert len(traj1) == len(traj2)

    aligned_traj2 = [traj1[0]]
    for i in range(len(traj2)-1):
        odom = traj2[i].between(traj2[i+1])
        aligned_traj2.append(aligned_traj2[-1].compose(odom))

    return aligned_traj2

def read_trajectory(path):
    pose_dict = {}
    pose_keys = []
    file = open(path, 'r')
    for line in file.readlines():
        if line[0] != '#':
            line = list(map(float, line.strip().split(" ")))
            key = str(line[0])
            pose_dict[key] = gtsam.Pose3(gtsam.Rot3.Quaternion(line[-1], *line[4:-1]), gtsam.Point3(*line[1:4]))
            pose_keys.append(key)
    
    return pose_keys, pose_dict


