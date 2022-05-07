import argparse
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import wandb
from config import Config

from slam_solver import *
from drawing import Visualizer
from instances import Instances
from evaluation import Evaluation

import warnings

warnings.filterwarnings("ignore")


def main(config_path: str) -> None:
    # Seeding
    np.random.seed(0)
    random.seed(0)
    # load config
    config = yaml.safe_load(open(config_path, 'r'))
    config = Config(config)

    # load dataset
    dir_path = config.dir
    instances = Instances.load_dataset(dir_path)  # Load with orb trajectories.

    # visualizer
    visualizer = Visualizer(instances.cam_ids, instances.bbox_ids, instances.calibration)

    # Slam system
    # slam = SLAM(instances.calibration, config)
    slam = Calib_SLAM(instances.calibration, config)
    # slam = QuadricSLAM(instances.calibration, config)
    initial_estimates = slam.make_graph(instances)
    results = slam.solve(initial_estimates)

    # Evaluation
    ground_truth = instances.get_gt()
    orb_trajectory = instances.toValues()
    metrics = slam.evaluate(ground_truth, orb_trajectory)
    print('Ground truth vs ORB estimate: {}'.format(metrics))
    metrics = slam.evaluate(initial_estimates, results)
    print('Initial estimate vs Final results: {}'.format(metrics))
    metrics = slam.evaluate(ground_truth, results)
    print('Ground truth vs Final estimate: {}'.format(metrics))

    # print("-------Visualizing----------")
    print("Visualizing")
    visualizer.plot_comparison(instances.toValues(), initial_estimates,
                               "ORB vs Init", add_landmarks=config.add_landmarks)
    visualizer.plot_comparison(instances.toValues(), results,
                               "ORB vs Estimated", add_landmarks=config.add_landmarks)
    fig = visualizer.fig
    plt.show()
    visualizer.visualize(instances, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Batch Quadrics SLAM')

    parser.add_argument("--config_path", '-c', type=str, action='store', help="Path to YAML config file",
                        default='fr2_config.yaml', required=True)

    args = parser.parse_args()
    main(**vars(args))
