import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import wandb
from config import Config

from slam_solver import *
from drawing import Visualizer
import dataloader

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)


# load config
config_path = sys.argv[1]
config = yaml.safe_load(open(config_path, 'r'))
config = Config(config)

# load dataset
dir_path = config.dir
file = open(dir_path+"calibration.yaml", 'r')
intrinsics  = yaml.load(file)
instances = dataloader.tum_orb(dir_path, intrinsics) # Load with orb trajectories.

# visualizer
visualizer = Visualizer(instances.cam_ids, instances.bbox_ids, instances.calibration)

# slam system
slam = SLAM(intrinsics, config)
initial_estimates = slam.make_graph(instances)
results = slam.solve(initial_estimates)

# evaluation
metrics = slam.evaluate(instances.toValues(), results)
print(metrics)

# print("-------Visualizing----------")
visualizer.plot_comparison(instances.toValues(), initial_estimates, "GT vs Init", add_landmarks = config.add_landmarks)
# visualizer.plot_comparison(instances.toValues(), results, "Init vs Estimated", add_landmarks=config.add_landmarks) 

fig = visualizer.fig

plt.show()




