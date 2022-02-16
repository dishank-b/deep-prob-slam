import numpy as np
from scipy.optimize import linear_sum_assignment

class Evaluation(object):
    @staticmethod
    def ATE(trajectory1, trajectory2):
        """
        Compute the ATE (Average Translation Error) between
        the two trajectories.
        """
        error = [p1.between(p2).translation() for p1, p2 in zip(trajectory1, trajectory2)]
        rmse = np.sqrt(np.mean(error**2))
        return rmse

