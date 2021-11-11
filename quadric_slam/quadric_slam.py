import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import yaml
import wandb

import slam_solver
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
            group="calibration",
            config = {
            "odom_sigma" : [2, 0.01],
            "bbox_sigma" : 5, 
            }
        )

        wandb.run.name = str(run)

        config = wandb.config

        PRIOR_SIGMA = [1*np.pi/180]*3 + [1e-4]*3
        ODOM_SIGMA = [config.odom_sigma[0]*np.pi/180]*3 + [config.odom_sigma[1]]*3  # reasonable range angle = 10-15˚, translation = 10-20cm
        BOX_SIGMA = [config.bbox_sigma]*4
        landmarks = True


        # instances = dataloader.tum_raw("./data/preprocessed/")
        file = open("./data/calibration.yaml", 'r')
        intrinsics  = yaml.load(file)
        instances = dataloader.tum_uncertainty("./data/tum.pth", intrinsics)

        print("-------DATA LOADED--------------")
        # slam = slam_solver.SLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA, BOX_SIGMA)
        slam = slam_solver.Calib_SLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA)

        print("-------Making graph--------------")
        initial_estimates = slam.make_graph(instances, add_landmarks=landmarks)
        
        print("-------Solving graph--------------")
        results = slam.solve(initial_estimates)
        
        print("-------Evaluating--------------")
        metrics = slam.evaluate(initial_estimates, results)
        print(metrics)

        print("-------Visualizing----------")
        visualizer = drawing.Visualizer(slam.cam_ids, slam.bbox_ids, slam.calibration)
        visualizer.plot_comparison(initial_estimates, results, "Init vs Estimated", add_landmarks=landmarks)
        visualizer.plot_comparison(instances.toValues(), initial_estimates, "GT vs Init", add_landmarks = landmarks)

        wandb.log(metrics)

        wandb.finish()

if __name__ == "__main__":
    main()



