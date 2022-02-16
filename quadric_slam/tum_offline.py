import sys
import numpy as np
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

np.random.seed(0)


# load config
config_path = sys.argv[1]
config = yaml.safe_load(open(config_path, 'r'))
config = Config(config)

# load dataset
dir_path = config.dir
instances = Instances.load_dataset(dir_path) # Load with orb trajectories.

# visualizer
visualizer = Visualizer(instances.cam_ids, instances.bbox_ids, instances.calibration)

# slam system
# slam = SLAM(instances.calibration, config)
slam = Calib_SLAM(instances.calibration, config)
# slam = QuadricSLAM(instances.calibration, config)
initial_estimates = slam.make_graph(instances)
results = slam.solve(initial_estimates)

# evaluation
metrics = slam.evaluate(initial_estimates, results)
print(metrics)
metrics = slam.evaluate(instances.get_gt(), results)
print(metrics)

# print("-------Visualizing----------")
print("visualizign")
visualizer.plot_comparison(instances.toValues(), initial_estimates, "GT vs Init", add_landmarks = config.add_landmarks)
visualizer.plot_comparison(instances.toValues(), results, "Init vs Estimated", add_landmarks=config.add_landmarks) 

fig = visualizer.fig

plt.show()

visualizer.visualize(instances, results)



