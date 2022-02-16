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


path = "./data/fr3_office/"

orb_file = open(path+"CameraTrajectory_ORBVO.txt", 'r')
gt_file = open(path+"groundtruth.txt", 'r')

orb_dict = {}
gt_dict = {}
orb_keys = []
gt_keys = []

for line in orb_file.readlines():
    split_line = list(map(float, line.strip().split(" ")))
    str_name = str(split_line[0])
    orb_dict[str_name] =  gtsam.Pose3(gtsam.Rot3.Quaternion(split_line[-1], *split_line[4:-1]), gtsam.Point3(*split_line[1:4]))
    orb_keys.append(str_name)

for line in gt_file.readlines():
    if line[0] != '#':
        split_line = list(map(float, line.strip().split(" ")))
        str_name = str(split_line[0])
        gt_dict[str_name] = gtsam.Pose3(gtsam.Rot3.Quaternion(split_line[-1], *split_line[4:-1]), gtsam.Point3(*split_line[1:4]))
        gt_keys.append(str_name)

matches = align_times(orb_keys, gt_keys)

print(len(orb_keys))
orb = {}
gt = {}
orb_keys = []
gt_keys = []
for (orbkey, gtkey) in matches:
    orb[orbkey] = orb_dict[orbkey]
    gt[gtkey] = gt_dict[gtkey]
    orb_keys.append(orbkey)
    gt_keys.append(gtkey)

print(len(orb_keys))

gt_pose = [orb[orb_keys[0]]]
for i in range(len(orb_keys)-1):
    odom = gt[gt_keys[i]].between(gt[gt_keys[i+1]])
    gt_pose.append(gt_pose[-1].compose(odom))

orb_pose = []
for key in orb_keys:
    orb_pose.append(orb[key])

ate = []
for orb, gt in zip(orb_pose, gt_pose):
    ate.append(np.linalg.norm(orb.between(gt).translation()))
ate = np.array(ate)
ate = np.sqrt(np.average(ate**2))

print(ate)
