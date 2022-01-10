import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import yaml
import wandb

from slam_solver import *
import drawing
import dataloader

import warnings
warnings.filterwarnings("ignore")

def main():

    runs = 1

    np.random.seed(0)

    for run in range(runs):
        wandb.init(
            mode="disabled",
            project="quadric_slam",
            entity="deeprobslam",
            group="odom-noisy-box-noisy-gtquad-sweep-init",
            config = {
            "odom_sigma" : [2, 0.01],
            # "odom_sigma" : [0.06, 0.001],
            "bbox_sigma" : 5, 
            }
        )

        wandb.run.name = str(run)

        config = wandb.config

        PRIOR_SIGMA = [0.01*np.pi/180]*3 + [1e-4]*3
        ODOM_SIGMA = [config.odom_sigma[0]*np.pi/180]*3 + [config.odom_sigma[1]]*3  # reasonable range angle = 10-15˚, translation = 10-20cm
        QUADSLAM_ODOM_SIGMA = [0.001]*6  # QuadricSLAM hyper parameter
        BOX_SIGMA = [config.bbox_sigma]*4
        landmarks = True
        add_odom_noise = True
        add_meas_noise = True

        file = open("./data/calibration.yaml", 'r')
        intrinsics  = yaml.load(file)
        # instances = dataloader.tum_raw("./data/preprocessed/", intrinsics)
        instances = dataloader.tum_uncertainty("./data/tum.pth", intrinsics)        # calibrated uncertainty
        # instances = dataloader.tum_uncertainty("./data/tum_nll.pth", intrinsics)  # nll undertainty
        # instances = instances[:2000]
        visualizer = drawing.Visualizer(instances.cam_ids, instances.bbox_ids, instances.calibration)
        print("-------DATA LOADED--------------")
        
        
        # slam = SLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA, BOX_SIGMA)
        # slam = Calib_SLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA)
        # slam = QuadricSLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA)
        slam = IncrementalSLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA, BOX_SIGMA)

        print("-------Making graph--------------")
        # initial_estimates = slam.make_graph(instances, add_landmarks=landmarks, add_odom_noise=add_odom_noise, add_meas_noise=add_meas_noise)
        # visualizer.plot_comparison(instances.toValues(), initial_estimates, "GT vs Init", add_landmarks = landmarks)

        # plt.show()
        print("-------Solving graph--------------")
        # results = slam.solve(initial_estimates)
        results = slam.solve(instances, add_odom_noise) # to be used with IncrementalSLAM

        print("-------Evaluating--------------")
        # init_metrics = slam.evaluate(instances.toValues(), initial_estimates)
        # print(init_metrics)
        # metrics = slam.evaluate(initial_estimates, results)
        metrics = slam.evaluate(instances.toValues(), results)
        print(metrics)

        # print("-------Visualizing----------")
        visualizer.plot_comparison(instances.toValues(), results, "Init vs Estimated", add_landmarks=landmarks) 

        fig = visualizer.fig

        plt.show()

        # wandb.log(metrics | {"Trajectory": wandb.Image(fig)})

        # wandb.finish()

if __name__ == "__main__":
    main()



