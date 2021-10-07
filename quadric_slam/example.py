import sys
import numpy as np

import gtsam
import gtsam_quadrics

print(gtsam.__file__, gtsam_quadrics.__file__)
# setup constants
pose_key = int(gtsam.symbol('x', 0))
quadric_key = int(gtsam.symbol('q', 5))

# create calibration
# This is camera intrinsic parameters, basically K in the equation P = K[R|t] where P is camera projection matrix
calibration = gtsam.Cal3_S2(525.0, 525.0, 0.0, 160.0, 120.0)

# create graph/values
graph = gtsam.NonlinearFactorGraph()
values = gtsam.Values()

# create noise model (SD=10)
bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2]*4, dtype=float))

# create quadric landmark (pose=eye(4), radii=[1,2,3])
initial_quadric = gtsam_quadrics.ConstrainedDualQuadric(gtsam.Pose3(), np.array([1.,1.,1.]))

# create bounding-box measurement (xmin,ymin,xmax,ymax)
bounds = gtsam_quadrics.AlignedBox2(100,100,120,120)
 
# create bounding-box factor
bbf = gtsam_quadrics.BoundingBoxFactor(bounds, calibration, pose_key, quadric_key, bbox_noise)

# add landmark to values
initial_quadric.addToValues(values, quadric_key)
values.insert(pose_key, gtsam.Pose3())

# add bbf to graph
graph.add(gtsam.PriorFactorPose3(pose_key, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1*np.pi/180, 1*np.pi/180, 1*np.pi/180, 0.05, 0.05, 0.05]))))
graph.add(bbf)

#solve
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
result = optimizer.optimize()

# get quadric estimate from values (assuming the values have changed)
quadric_estimate = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(values, quadric_key)

print(quadric_estimate)