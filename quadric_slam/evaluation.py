import numpy as np
from scipy.optimize import linear_sum_assignment

import gtsam
import gtsam_quadrics
from gtsam.symbol_shorthand import X, L


class Evaluation(object):

    def __init__(self, cam_ids):
        self.cam_ids = cam_ids

    @staticmethod
    def ATE(trajectory1, trajectory2):
        """ point error = || x ominus x* || """
        assert len(trajectory1) == len(trajectory2)
        trans_errors = []
        for p1, p2 in zip(trajectory1, trajectory2):
            point_error = np.linalg.norm(p1.between(p2).translation())
            trans_errors.append(point_error)
        trans_errors = np.array(trans_errors)
        rmse = np.sqrt(np.average(trans_errors ** 2))
        return rmse

    @staticmethod
    def RPE(trajectory1, trajectory2):
        """
        Alignes each pose p and calculates the ate between the next poses
        point error = || rp ominus rp* ||
        """
        assert len(trajectory1) == len(trajectory2)
        trans_errors = []
        for i, j in zip(range(len(trajectory1) - 1), range(1, len(trajectory1))):
            ap1 = trajectory1[i].between(trajectory1[j])
            ap2 = trajectory2[i].between(trajectory2[j])
            pose_diff = ap1.between(ap2)
            trans_errors.append(np.linalg.norm(pose_diff.translation()))
        trans_errors = np.array(trans_errors)
        rmse = np.sqrt(np.average(trans_errors ** 2))
        return rmse

    @staticmethod
    def apply_transform(trajectory, mapping):
        transformed_trajectory = []
        for pose in trajectory:
            transformed_trajectory.append(mapping.transformPoseFrom(pose))
        return transformed_trajectory

    def align_trajectories(self, estimated_trajectory, true_trajectory, type):
        true_trajectory = [true_trajectory.atPose3(X(idx)) for idx in self.cam_ids]
        estimated_trajectory = [estimated_trajectory.atPose3(X(idx)) for idx in self.cam_ids]
        # align trajectories
        transform = Evaluation.calculate_transform(estimated_trajectory, true_trajectory, type=type)
        aligned_trajectory = Evaluation.apply_transform(estimated_trajectory, transform)
        return true_trajectory, aligned_trajectory

    def evaluate_trajectory(self, estimated_trajectory, true_trajectory, type):
        # Align trajectories
        true_trajectory, aligned_trajectory = self.align_trajectories(estimated_trajectory, true_trajectory, type)

        # evaluate metrics
        rmse_ATE = Evaluation.ATE(aligned_trajectory, true_trajectory)
        rmse_RPE = Evaluation.RPE(aligned_trajectory, true_trajectory)
        return rmse_ATE, rmse_RPE

    @staticmethod
    def calculate_transform(trajectory1, trajectory2, type):
        if type == 'horn':
            xyz1 = np.matrix([p.translation() for p in trajectory1]).transpose()
            xyz2 = np.matrix([p.translation() for p in trajectory2]).transpose()
            R, T, trans_error = Evaluation._horn_align(xyz1, xyz2)
            transform = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(np.array(T)[:, 0]))
        elif type == 'weak':
            transform = trajectory1[0].between(trajectory2[0])
        return transform

    @staticmethod
    def _horn_align(model, data):
        """Align two trajectories using the method of Horn (closed-form).

        Input:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

        Output:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

        From: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools

        """
        model_zerocentered = model - model.mean(1)
        data_zerocentered = data - data.mean(1)

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if np.linalg.det(U) * np.linalg.det(Vh) < 0:
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1) - rot * model.mean(1)

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error
