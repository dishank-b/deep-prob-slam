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
    # Initialize evaluator
    eval = Evaluation(instances.cam_ids)
    ate, rpe = eval.evaluate_trajectory(ground_truth, orb_trajectory, type='horn')
    print('Ground truth vs ORB estimate ATE: {}'.format(ate))
    ate, rpe = eval.evaluate_trajectory(ground_truth, initial_estimates, type='horn')
    print('Ground truth vs Initial estimate ATE: {}'.format(ate))
    ate, rpe = eval.evaluate_trajectory(ground_truth, results, type='horn')
    print('Ground truth vs Final estimate ATE: {}'.format(ate))
    # metrics = slam.evaluate(ground_truth, orb_trajectory)
    # print('Ground truth vs ORB estimate: {}'.format(metrics))
    # metrics = slam.evaluate(ground_truth, initial_estimates)
    # print('Ground truth vs Initial estimate: {}'.format(metrics))
    # metrics = slam.evaluate(ground_truth, results)
    # print('Ground truth vs Final estimate: {}'.format(metrics))

    # print("-------Visualizing----------")
    print("Visualizing")
    # Ground truth vs ORB
    visualizer.reset_figure((10, 8))
    visualizer.plot_comparison(ground_truth, orb_trajectory,
                               "GT vs ORB", add_landmarks=False, labels=['GT', 'ORB'])
    # Plot first Figure and reset figure
    fig = visualizer.fig
    plt.savefig('results/gt_vs_orb.png')
    # # Initial estimate vs Final estimate
    visualizer.reset_figure()
    visualizer.plot_comparison(orb_trajectory, results, "ORB vs Final",
                               add_landmarks=config.add_landmarks, labels=['ORB', 'Final'])
    # Plot first Figure and reset figure
    fig = visualizer.fig
    plt.savefig('results/orb_vs_final.png')
    # ORB vs Final estimate
    visualizer.reset_figure((10, 8))
    visualizer.plot_comparison(ground_truth, results,
                               "ORB vs Estimated", add_landmarks=False, labels=['GT', 'Final'])
    fig = visualizer.fig
    plt.savefig('results/gt_vs_final.png')
    # ORB vs Initialization
    visualizer.reset_figure()
    visualizer.plot_comparison(orb_trajectory, initial_estimates,
                               "ORB vs Init", add_landmarks=config.add_landmarks, labels=['ORB', 'Noisy'])
    fig = visualizer.fig
    fig.tight_layout()
    plt.savefig('results/orb_vs_noisy.png')
    visualizer.visualize(instances, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Batch Quadrics SLAM')

    parser.add_argument("--config_path", '-c', type=str, action='store', help="Path to YAML config file",
                        default='fr2_config.yaml', required=True)

    args = parser.parse_args()
    main(**vars(args))
