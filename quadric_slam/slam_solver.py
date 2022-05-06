import numpy as np
import cv2

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L
# X = lambda i: int(gtsam.symbol(ord('x'), i))
# L = lambda i: int(gtsam.symbol(ord('l'), i))

from instances import Instances
from quadrics_multiview import initialize_quadric
from drawing import CV2Drawing



class SLAM(object):
    """
    Class for solve the object slam system 
    """
    def __init__(self, calibration, config) -> None:
        super().__init__()
        self.graph = self._init_graph()
        self.calibration = calibration
        
        prior_sigma = [config.prior_sigma[0]*np.pi/180]*3 + [config.prior_sigma[1]]*3
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(prior_sigma, dtype=float))
        odom_sigma = [config.odom_sigma[0]*np.pi/180]*3 + [config.odom_sigma[1]]*3  # reasonable range angle = 10-15˚, translation = 10-20cm
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(odom_sigma, dtype=float))
        bbox_sigma = [config.bbox_sigma]*4
        self.bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(bbox_sigma, dtype=float))
        
        self.cam_ids = []
        self.bbox_ids = []

        self.add_landmarks = config.add_landmarks
        self.add_odom_noise = config.add_odom_noise
        self.add_meas_noise = config.add_measurement_noise

    def _init_graph(self):
        """
        Initialize the graph
        TODO: add an argument to chose which type of graph to use
        """
        return gtsam.NonlinearFactorGraph()

    def _add_landmark(self, instance, bbox_stds, add_noise=True):
        for obj_id, bbox in zip(instance.object_key, instance.bbox):
            if add_noise:
                bbox = np.random.multivariate_normal(bbox, self.bbox_noise.covariance())
            box = gtquadric.AlignedBox2(*bbox)
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), self.bbox_noise)
            self.graph.add(bbf)

    def make_graph(self, instances: Instances):
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
                if self.add_odom_noise:
                    noise = np.random.multivariate_normal(np.zeros(6), self.odometry_noise.covariance())
                    relative_pose_true = instance.pose.between(pose_t1)
                    pose_t1 = instance.pose.compose(relative_pose_true.compose(pose_t1.Expmap(noise)))
                relative_pose = instance.pose.between(pose_t1)
                odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i+1].image_key), relative_pose, self.odometry_noise)
                # odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i+1].image_key), relative_pose_true, self.odometry_noise)
                self.graph.add(odometry_factor)
                initial_estimate.insert(X(instances[i+1].image_key), pose_t1)
                # initial_estimate.insert(X(instances[i+1].image_key), initial_estimate.atPose3(X(image_key)).compose(relative_pose))

            if self.add_landmarks:
                quadric_slam_stds = instances.get_bbox_std() # Std dev used in QuadricSLAM paper
                self._add_landmark(instance, quadric_slam_stds, self.add_meas_noise)
        
        if self.add_landmarks:
            gt_values = instances.toValues()
            self.bbox_ids = instances.bbox_ids
            for box_id in self.bbox_ids:
                quadric = gtquadric.ConstrainedDualQuadric.getFromValues(gt_values, L(box_id))
                quadric.addToValues(initial_estimate, L(box_id))

        return initial_estimate

    def solve(self, initial_estimate):
        """
        Optimization of factor graph
        """
        # define lm parameters
        parameters = gtsam.LevenbergMarquardtParams()
        # parameters = gtsam.DoglegParams()
        # parameters = gtsam.GaussNewtonParams() 
        
        parameters.setVerbosityLM("SILENT") # SILENT = 0, SUMMARY, TERMINATION, LAMBDA, TRYLAMBDA, TRYCONFIG, DAMPED, TRYDELTA : VALUES, ERROR 
        parameters.setMaxIterations(100)
        parameters.setlambdaInitial(1e-5)
        parameters.setlambdaUpperBound(1e10)
        parameters.setlambdaLowerBound(1e-8)
        parameters.setRelativeErrorTol(1e-5)
        parameters.setAbsoluteErrorTol(1e-5)
        # parameters.setFactorization("QR")

        # create optimizer
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, initial_estimate, parameters)
        # optimizer = gtsam.DoglegOptimizer(self.graph, initial_estimate, parameters)
        # optimizer = gtsam.GaussNewtonOptimizer()(self.graph, initial_estimate, parameters)

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
        cam_rmse = np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(gt_cam, esti_cam)]).mean()
        metrics.update({"Cam pose RMSE": cam_rmse})

        # init_quad = [gtquadric.ConstrainedDualQuadric.getFromValues(gt, L(id)).centroid() for id in self.bbox_ids]
        # esti_quad = [gtquadric.ConstrainedDualQuadric.getFromValues(results, L(id)).centroid() for id in self.bbox_ids]
        # quad_rmse = np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(init_quad, esti_quad)]).mean()
        # metrics.update({"Quadrics RMSE": quad_rmse})

        return metrics

class Calib_SLAM(SLAM):
    def __init__(self,calibration, config) -> None:
        super().__init__(calibration, config)

    def _add_landmark(self, instance, bbox_stds, add_noise=False):
        for obj_id, bbox, bbox_covar in zip(instance.object_key, instance.bbox, instance.bbox_covar):
            if add_noise:
                bbox = np.random.multivariate_normal(bbox, bbox_covar)
            box = gtquadric.AlignedBox2(*bbox)
            bbox_noise = gtsam.noiseModel.Gaussian.Covariance(np.array(bbox_covar, dtype=float))
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), bbox_noise)
            self.graph.add(bbf)

class QuadricSLAM(SLAM):
    def __init__(self,calibration, config) -> None:
        super().__init__(calibration, config)

    def _add_landmark(self, instance, bbox_stds, add_noise=False):
        for obj_id, bbox in zip(instance.object_key, instance.bbox):
            bbox_noise = bbox_stds[obj_id]
            if add_noise:
                bbox = np.random.multivariate_normal(bbox, bbox_noise.covariance())
            box = gtquadric.AlignedBox2(*bbox)
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), bbox_noise)
            self.graph.add(bbf)

class IncrementalSLAM(SLAM):
    def __init__(self, calibration, prior_sigma, odom_sigma, bbox_sigma=[20] * 4) -> None:
        super().__init__(calibration, prior_sigma, odom_sigma, bbox_sigma=bbox_sigma)
        self.isam = self.optimizer()
        self.local_estimate = gtsam.Values()

    def make_graph(self, instances: Instances, add_landmarks=True, add_odom_noise=True, add_meas_noise=True):
        raise AttributeError

    def optimizer(self):
        # create isam optimizer 
        opt_params = gtsam.ISAM2DoglegParams()
        params = gtsam.ISAM2Params()
        params.setOptimizationParams(opt_params)
        params.setEnableRelinearization(True)
        params.setRelinearizeThreshold(0.1)
        params.setRelinearizeSkip(1)
        params.setCacheLinearizedFactors(False)
        
        isam = gtsam.ISAM2(params)  

        return isam

    def solve(self, instances, add_odom_noise):
        # create storage for traj/map estimates
        current_trajectory = {}
        current_quadrics = {}

        # store quadrics until they have been viewed enough times to be constrained (>3)
        unconstrained_quadrics = {}

        step = 1

        gt_values = instances.toValues()

        for i, instance in enumerate(instances):
            if i==0:
                curr_key = instance.image_key
                curr_pose = instance.pose
                prior_factor = gtsam.PriorFactorPose3(X(curr_key), curr_pose, self.prior_noise)
                self.local_estimate.insert(X(curr_key), curr_pose)
                self.graph.add(prior_factor)
            
            else:                
                curr_pose = instance.pose
                curr_key = instance.image_key
                previous_pose = instances[i-1].pose
                if add_odom_noise:
                    noise = np.random.multivariate_normal(np.zeros(6), self.odometry_noise.covariance())
                    relative_pose_true = previous_pose.between(curr_pose)
                    curr_pose = previous_pose.compose(relative_pose_true.compose(curr_pose.Expmap(noise)))
                odom = previous_pose.between(curr_pose)

                # compound odometry to global pose
                previous_pose = current_trajectory[prev_key]
                curr_pose = previous_pose.compose(odom) # current pose in world frame

                # add pose estimate to values and current estimateo (for initialization)
                self.local_estimate.insert(X(curr_key), curr_pose) 

                # add odometry factor to graph
                odom_factor = gtsam.BetweenFactorPose3(X(prev_key), X(curr_key), odom, self.odometry_noise)
                self.graph.add(odom_factor) 

                # print("add factor betweeen {} and {}".format(prev_key, curr_key))
            
            prev_key = curr_key

            self.cam_ids.append(curr_key)
            
            current_trajectory[curr_key] = curr_pose


            boxes = instance.bbox # bbox of current frame
            # associate boxes -> quadrics 
            associated_keys = instance.object_key
            # print(associated_keys)
            # wrap boxes with keys 
            associated_boxes = []
            for box, quadric_key in zip(boxes, associated_keys):
                associated_boxes.append({
                    'box': box,
                    'quadric_key': quadric_key,
                    'pose_key': curr_key,
                })

            unconstrained_quadrics_keys = unconstrained_quadrics.keys()

            # initialize new landmarks
            new_boxes = [f for f in associated_boxes if f['quadric_key'] not in current_quadrics.keys()]
            old_boxes = [f for f in associated_boxes if f['quadric_key'] in current_quadrics.keys()]
            for detection in new_boxes:
                quadric_key = detection['quadric_key']
                if quadric_key in unconstrained_quadrics_keys:
                    unconstrained_quadrics[quadric_key].append(detection)
                else:
                    unconstrained_quadrics[quadric_key] = [detection]

            # add initialized landmarks to values (if constrained)
            temp_dir = unconstrained_quadrics.copy()
            for quadric_key in unconstrained_quadrics_keys:
                quadric_measurements = unconstrained_quadrics[quadric_key]
                if len(quadric_measurements) > 3:
                    # quadric = initialize_quadric(quadric_measurements, current_trajectory, self.calibration)
                    quadric = gtquadric.ConstrainedDualQuadric.getFromValues(gt_values, L(quadric_key))
                    quadric.addToValues(self.local_estimate, L(quadric_key))
                    current_quadrics[quadric_key] = quadric
                    for measurement in quadric_measurements:
                        box = measurement['box']
                        box = gtquadric.AlignedBox2(*box)
                        quadric_key = measurement['quadric_key']
                        pose_key = measurement['pose_key']
                        bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key), self.bbox_noise, "STANDARD")
                        self.graph.add(bbf)
                    temp_dir.pop(quadric_key)
            unconstrained_quadrics = temp_dir
            # add measurements to graph if quadric is initialized and constrained
            # current_quadrics_keys = current_quadrics.keys()
            for detection in old_boxes:
                box = detection['box']
                box = gtquadric.AlignedBox2(*box)
                quadric_key = detection['quadric_key']
                pose_key = detection['pose_key']
                bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key), self.bbox_noise, "STANDARD")
                self.graph.add(bbf)
            
            # print(current_quadrics)
            # add local graph and estimate to isam
            # print(self.graph)
           
            # draw the current object detections
            if len(current_quadrics) != 0:
                image_path = instance.image_path
                image_path = "/".join(["data"] + image_path.split('/')[-2:])
                image = cv2.imread(image_path)
                drawing = CV2Drawing(image)
                for frame in associated_boxes:
                    text = '{}'.format(frame['quadric_key'])
                    drawing.box_and_text(gtquadric.AlignedBox2(*frame['box']), (0,0,255), text, (0,0,0))

                # visualize current view into map
                camera_pose = current_trajectory[max(current_trajectory.keys())]
                for quadric in current_quadrics.values():
                    drawing.quadric(camera_pose, quadric, self.calibration, (255,0,255))
                cv2.imshow('current view', image)
                cv2.waitKey(0)

            
           
            self.isam.update(self.graph, self.local_estimate)
            estimate = self.isam.calculateEstimate()

            # clear graph/estimate
            self.graph.resize(0)
            self.local_estimate.clear()

            # update the estimated quadrics and trajectory
            for i in range(len(estimate.keys())):
                key = estimate.keys()[i]
                if chr(gtsam.symbolChr(key)) == 'l':
                    quadric = gtquadric.ConstrainedDualQuadric.getFromValues(estimate, key)
                    current_quadrics[gtsam.symbolIndex(key)] = quadric
                elif chr(gtsam.symbolChr(key)) == 'x':
                    current_trajectory[gtsam.symbolIndex(key)] = estimate.atPose3(key)



            step += 1
            # print(estimate)

        return estimate

    def evaluate(self, gt, results):
        gt_cam = []
        esti_cam = []
        for i in range(len(results.keys())):
            key = results.keys()[i]
            if chr(gtsam.symbolChr(key)) == 'x':
                gt_cam.append(gt.atPose3(key).translation())
                esti_cam.append(results.atPose3(key).translation())
        cam_rmse = np.array([np.linalg.norm(gt_pose-esti_pose) for gt_pose, esti_pose in zip(gt_cam, esti_cam)]).mean()

        return {"Cam RMSE": cam_rmse}
        

        