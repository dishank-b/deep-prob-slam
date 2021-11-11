import numpy as np

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L

from instances import Instances

class SLAM(object):
    """
    Class for solve the object slam system 
    """
    def __init__(self, intrinsics, prior_sigma, odom_sigma, bbox_sigma = [20.0]*4) -> None:
        super().__init__()
        self.graph = self._init_graph()
        self.calibration = gtsam.Cal3_S2(intrinsics["fx"], intrinsics["fy"], 0.0, intrinsics["cx"], intrinsics["cy"])
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(prior_sigma, dtype=float))
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(odom_sigma, dtype=float))
        self.bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(bbox_sigma, dtype=float))
        self.cam_ids = []
        self.bbox_ids = []

    def _init_graph(self):
        """
        Initialize the graph
        TODO: add an argument to chose which type of graph to use
        """
        return gtsam.NonlinearFactorGraph()

    def _add_landmark(self, instance, add_noise=True):
        for obj_id, bbox in zip(instance.object_key, instance.bbox):
            if add_noise:
                bbox = np.random.multivariate_normal(bbox, self.bbox_noise.covariance())
            box = gtquadric.AlignedBox2(*bbox)
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), self.bbox_noise)
            self.graph.add(bbf)

    def make_graph(self, instances: Instances, add_landmarks = True, add_odom_noise=True, add_meas_noise=True):
        """
        Make the factor graph to be solved.
        Data association is assumed to be solved for this. 
        No incremental solving. Joint optimizaition for all the camera poses and object poses at once. 

        Uses grounth truth odometry as the odometry measurements. 
        """
        initial_estimate = gtsam.Values()
        
        for i, instance in enumerate(instances):
            image_key = instance.image_key
            self.cam_ids.append(image_key)
            if i==0:
                self.graph.add(gtsam.PriorFactorPose3(X(image_key), instance.pose, self.prior_noise))
                initial_estimate.insert(X(image_key), gtsam.Pose3(instance.pose))
            
            if i < len(instances)-1:
                pose_t1 = instances[i+1].pose
                if add_odom_noise:
                    noise = np.random.multivariate_normal(np.zeros(6), self.odometry_noise.covariance())
                    relative_pose = instance.pose.between(pose_t1)
                    pose_t1 = instance.pose.compose(relative_pose.compose(pose_t1.Expmap(noise)))
                relative_pose = instance.pose.between(pose_t1)
                odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i+1].image_key), relative_pose, self.odometry_noise)
                self.graph.add(odometry_factor)
                initial_estimate.insert(X(instances[i+1].image_key), initial_estimate.atPose3(X(image_key)).compose(relative_pose))

            if add_landmarks:
                self._add_landmark(instance, add_meas_noise)
        
        if add_landmarks:
            gt_values = instances.toValues()
            self.bbox_ids = instances.get_box_ids()
            for box_id in instances.get_box_ids():
                quadric = gtquadric.ConstrainedDualQuadric.getFromValues(gt_values, L(box_id))
                quadric.addToValues(initial_estimate, L(box_id))

        return initial_estimate

    def solve(self, initial_estimate):
        """
        Optimization of factor graph
        """
        # define lm parameters
        parameters = gtsam.LevenbergMarquardtParams()
        parameters.setVerbosityLM("SILENT") # SILENT = 0, SUMMARY, TERMINATION, LAMBDA, TRYLAMBDA, TRYCONFIG, DAMPED, TRYDELTA : VALUES, ERROR 
        parameters.setMaxIterations(100)
        parameters.setlambdaInitial(1e-5)
        parameters.setlambdaUpperBound(1e10)
        parameters.setlambdaLowerBound(1e-8)
        parameters.setRelativeErrorTol(1e-5)
        parameters.setAbsoluteErrorTol(1e-5)

        # create optimizer
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, initial_estimate, parameters)

        # run optimizer
        results = optimizer.optimize()

        return results

    def evaluate(self, gt, results):
        """
        Evaluation metrices 

        Returns:
            metrics: dict
        """

        metrics = {}
        gt_cam = [gt.atPose3(X(id)).translation() for id in self.cam_ids]
        esti_cam = [results.atPose3(X(id)).translation() for id in self.cam_ids]

        init_quad = [gtquadric.ConstrainedDualQuadric.getFromValues(gt, L(id)).centroid() for id in self.bbox_ids]
        esti_quad = [gtquadric.ConstrainedDualQuadric.getFromValues(results, L(id)).centroid() for id in self.bbox_ids]

        cam_rmse = np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(gt_cam, esti_cam)]).mean()
        quad_rmse = np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(init_quad, esti_quad)]).mean()

        metrics.update({"Cam pose RMSE": cam_rmse})
        metrics.update({"Quadrics RMSE": quad_rmse})

        return metrics

class Calib_SLAM(SLAM):
    def __init__(self, intrinsics, prior_sigma, odom_sigma) -> None:
        super().__init__(intrinsics, prior_sigma, odom_sigma)

    def _add_landmark(self, instance, add_noise=False):
        for obj_id, bbox, bbox_covar in zip(instance.object_key, instance.bbox, instance.bbox_covar):
            if add_noise:
                bbox = np.random.multivariate_normal(bbox, bbox_covar)
            box = gtquadric.AlignedBox2(*bbox)
            bbox_noise = gtsam.noiseModel.Gaussian.Covariance(np.array(bbox_covar, dtype=float)) # is this right? need to check for this
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), bbox_noise)
            self.graph.add(bbf)