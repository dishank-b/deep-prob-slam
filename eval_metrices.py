import numpy as np
import scipy
from scipy import linalg
from scipy.linalg.basic import inv
from scipy.stats import norm
from scipy.stats import multivariate_normal
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def rmse(gt,estimate):
    return np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(gt, estimate)]).mean()

def mahalanobis(gt, estimate, var):
    return np.array([scipy.spatial.distance.mahalanobis(gt_pose, esti_pose, np.linalg.inv(covar)) for gt_pose, esti_pose, covar in zip(gt, estimate, var)]).mean()

def nll(gt, estimate, var):
    return np.array([-1*multivariate_normal.logpdf(gt_pose, esti_pose, covar) for gt_pose, esti_pose, covar in zip(gt, estimate, var)]).mean()

def ece(gt, estimate, var, plot=False):
    z = np.array([np.matmul(np.linalg.inv(scipy.linalg.sqrtm(covar)), x-u) for x, u, covar in zip(gt, estimate, var)])
    bin_size = 0.01
    conf_prev = 0
    ece = []
    for conf in np.arange(bin_size, 1+bin_size, bin_size):
        x_low = norm.ppf(conf_prev)
        x_upp = norm.ppf(conf)
        acc = np.logical_and(z > x_low, z <= x_upp)
        acc = acc.sum()/len(acc)
        ece.append(acc*np.abs(acc-bin_size))
        conf_prev = conf

    if plot:
        fig = plt.figure(0)
        axes = fig.gca()
        sns.distplot(z, kde=True, norm_hist=True)
        z_gt = np.arange(-4, 4, 0.01)
        axes.plot(z_gt, norm.pdf(z_gt), 'r', lw=2)

        wandb.log({"Calibration":wandb.Image(plt)})

        fig.clf()

    return np.array(ece).sum()

def odom_nll(gt_trajectory, esti_trajectory, covar):
    gt_odom = []
    esti_odom = []
    for idx in range(len(gt_trajectory)-1):
        diff = gt_trajectory[idx+1] - gt_trajectory[idx]
        d = np.linalg.norm(diff[:-1]) 
        delta = np.arctan2(diff[1],diff[0]) - gt_trajectory[idx][2]
        odometry = np.array([d*np.cos(delta), d*np.sin(delta), diff[2]])
        gt_odom.append(odometry)

        diff = esti_trajectory[idx+1] - esti_trajectory[idx]
        d = np.linalg.norm(diff[:-1]) 
        delta = np.arctan2(diff[1],diff[0]) - esti_trajectory[idx][2]
        odometry = np.array([d*np.cos(delta), d*np.sin(delta), diff[2]])
        esti_odom.append(odometry)

    gt_odom = np.asarray(gt_odom)
    esti_odom = np.asarray(esti_odom)

    return np.array([-1*multivariate_normal.logpdf(gt_pose, esti_pose, covar) for gt_pose, esti_pose in zip(gt_odom, esti_odom)]).mean()

