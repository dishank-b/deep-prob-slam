import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import yaml
import glob
import wandb

import gtsam 
from gtsam.symbol_shorthand import X, L
import gtsam_quadrics as gtquadric

import drawing
import dataloader

import warnings
warnings.filterwarnings("ignore")

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

    def _init_quad_single_frame(self, bbox, camera_pose, camera_intrinsics):
        """
        Calculate q given one boudning box and camera pose. 
        Solves for Aq = 0. Where A formed using one bouding box from the the image. 
        Rank of A (4) is less than number of variables in q (10) to solve properly/
        Hence underdeterminent system. 
        """
        planes = []
        
        for i in range(bbox.lines().size()):
            planes.append(gtquadric.QuadricCamera.transformToImage(camera_pose, camera_intrinsics).transpose()@bbox.lines().at(i))
        A = []
        
        for plane in planes:
            a = plane[..., None]*np.ones((len(plane), len(plane)))*plane
            A.append(a)
        
        for i in range(len(A)):
            a = A[i]
            a = np.triu(2*a)-np.diag(np.diag(a))
            A[i] = a[np.triu_indices(len(a))]
        
        A = np.array(A)
        
        return A

    def _init_quad_multi_frames(self, bboxes, camera_poses, camera_intrinsics):
        """
        Calculate quadric q given boudning box measurement over multiple frames and
        respective camera poses. 
        Solves for Aq = 0. Where A formed using one bouding box from the the image. 
        Rank of A (4) is less than number of variables in q (10) to solve properly/
        Hence underdeterminent system. 

        Refer to Eq. 9 in the paper
        """
        A = []
        
        for box, camera_pose in zip(bboxes, camera_poses):
            A.append(self._init_quad_single_frame(gtquadric.AlignedBox2(*box), camera_pose, camera_intrinsics))

        A = np.concatenate(A)
        
        _, _, VT = np.linalg.svd(A)
        q = VT.T[:,-1]
        Q = np.zeros((4,4))
        Q[np.triu_indices(4)] = q
        Q = Q+Q.T-np.diag(np.diag(Q))
        
        return gtquadric.ConstrainedDualQuadric(Q)

    def _init_quadrics(self, instances, initial_estimate):
        """
        Initialize the quadrics
        """
        obj_dict = {}
        
        for instance in instances:
            for obj_id, box in zip(instance.object_key, instance.bbox):
                if obj_id not in obj_dict:
                    obj_dict[obj_id] = [[box, instance.pose]]
                else:
                    obj_dict[obj_id].append([box, instance.pose])

        self.bbox_ids = sorted(list(obj_dict.keys()))
        
        for obj_id, box_cam_pair in obj_dict.items():
            quadric = self._init_quad_multi_frames(*zip(*box_cam_pair), self.calibration)
            quadric.addToValues(initial_estimate, L(obj_id))

    def make_graph(self, instances):
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
                relative_pose = instance.pose.between(instances[i+1].pose)
                # TODO: add noise to relative pose
                odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i+1].image_key), relative_pose, self.odometry_noise)
                self.graph.add(odometry_factor)
                initial_estimate.insert(X(instances[i+1].image_key), initial_estimate.atPose3(X(image_key)).compose(relative_pose))

            for obj_id, bbox in zip(instance.object_key, instance.bbox):
                box = gtquadric.AlignedBox2(*bbox)
                bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(image_key), L(obj_id), self.bbox_noise)
                self.graph.add(bbf)
                # if not initial_estimate.exists(L(obj_id)):
                #     self.obj_poses_key.append(obj_id)
                #     quadric = initialize_quadric(box, initial_estimate.atPose3(X(image_key)), self.calibration)
                #     # quadric = gtquadric.ConstrainedDualQuadric()
                #     quadric.addToValues(initial_estimate, L(obj_id))

        self._init_quadrics(instances, initial_estimate)

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

    def make_graph(self, instances):
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
                relative_pose = instance.pose.between(instances[i+1].pose)
                # TODO: add noise to relative pose
                odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i+1].image_key), relative_pose, self.odometry_noise)
                self.graph.add(odometry_factor)
                initial_estimate.insert(X(instances[i+1].image_key), initial_estimate.atPose3(X(image_key)).compose(relative_pose))

            for obj_id, bbox, bbox_covar in zip(instance.object_key, instance.bbox, instance.bbox_covar):
                box = gtquadric.AlignedBox2(*bbox)
                # bbox_noise = gtsam.noiseModel.Gaussian.Covariance(np.array(bbox_covar, dtype=float))
                # bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([20]*4, dtype=float))
                bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(image_key), L(obj_id), self.bbox_noise)
                self.graph.add(bbf)
                # if not initial_estimate.exists(L(obj_id)):
                #     self.obj_poses_key.append(obj_id)
                #     quadric = initialize_quadric(box, initial_estimate.atPose3(X(image_key)), self.calibration)
                #     # quadric = gtquadric.ConstrainedDualQuadric()
                #     quadric.addToValues(initial_estimate, L(obj_id))

        self._init_quadrics(instances, initial_estimate)

        return initial_estimate


def main():
    wandb.init(
        mode="disabled",
        project="quadric_slam",
        entity="deeprobslam",
        group="our_detection",
        config = {
        "odom_sigma" : [2, 0.01],
        "bbox_sigma" : 20, 
        }
    )

    config = wandb.config

    PRIOR_SIGMA = [1*np.pi/180]*3 + [1e-4]*3
    ODOM_SIGMA = [config.odom_sigma[0]*np.pi/180]*3 + [config.odom_sigma[1]]*3  # reasonable range angle = 10-15˚, translation = 10-20cm
    BOX_SIGMA = [config.bbox_sigma]*4

    #get data
    # instances = dataloader.tum_raw("./data/preprocessed/")
    instances = dataloader.tum_uncertainty("./data/data.pth")
    file = open("./data/calibration.yaml", 'r')
    intrinsics  = yaml.load(file)

    print("-------DATA LOADED--------------")
    slam = SLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA, BOX_SIGMA)
    # slam = Calib_SLAM(intrinsics, PRIOR_SIGMA, ODOM_SIGMA)
    print("-------Making graph--------------")
    initial_estimates = slam.make_graph(instances)
    print("-------Solving graph--------------")
    results = slam.solve(initial_estimates)
    print("-------Evaluating--------------")
    metrics = slam.evaluate(initial_estimates, results)
    print(metrics)

    visualizer = drawing.Visualizer(slam.cam_ids, slam.bbox_ids, slam.calibration)
    plots = visualizer.plot(initial_estimates, results, compare=False)
    # visualizer.visualize(instances, results)

    wandb.log(metrics | plots)

if __name__ == "__main__":
    main()



