import numpy as np
import cv2

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L

from instances import Instances
from quadrics_multiview import initialize_quadric
from drawing import CV2Drawing


class SLAM(object):
    """
    Class for solve the object slam system 
    """

    def __init__(self, intrinsics, prior_sigma, odom_sigma, bbox_sigma=[20.0] * 4) -> None:
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

    def make_graph(self, instances: Instances, add_landmarks=True, add_odom_noise=True, add_meas_noise=True):
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
            if i == 0:
                self.graph.add(gtsam.PriorFactorPose3(X(image_key), instance.pose, self.prior_noise))
                initial_estimate.insert(X(image_key), gtsam.Pose3(instance.pose))

            if i < len(instances) - 1:
                pose_t1 = instances[i + 1].pose
                if add_odom_noise:
                    noise = np.random.multivariate_normal(np.zeros(6), self.odometry_noise.covariance())
                    relative_pose_true = instance.pose.between(pose_t1)
                    pose_t1 = instance.pose.compose(relative_pose_true.compose(pose_t1.Expmap(noise)))
                relative_pose = instance.pose.between(pose_t1)
                odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i + 1].image_key), relative_pose,
                                                           self.odometry_noise)
                # odometry_factor = gtsam.BetweenFactorPose3(X(image_key), X(instances[i+1].image_key), relative_pose_true, self.odometry_noise)
                self.graph.add(odometry_factor)
                initial_estimate.insert(X(instances[i + 1].image_key), pose_t1)
                # initial_estimate.insert(X(instances[i+1].image_key), initial_estimate.atPose3(X(image_key)).compose(relative_pose))

            if add_landmarks:
                self._add_landmark(instance, add_meas_noise)

        if add_landmarks:
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
        parameters.setVerbosityLM(
            "SILENT")  # SILENT = 0, SUMMARY, TERMINATION, LAMBDA, TRYLAMBDA, TRYCONFIG, DAMPED, TRYDELTA : VALUES, ERROR
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

        cam_rmse = np.array(
            [np.linalg.norm(gt_pose - esti_pose) for gt_pose, esti_pose in zip(gt_cam, esti_cam)]).mean()
        quad_rmse = np.array(
            [np.linalg.norm(gt_pose - esti_pose) for gt_pose, esti_pose in zip(init_quad, esti_quad)]).mean()

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
            bbox_noise = gtsam.noiseModel.Gaussian.Covariance(np.array(bbox_covar, dtype=float))
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), bbox_noise)
            self.graph.add(bbf)


class QuadricSLAM(SLAM):
    def __init__(self, intrinsics, prior_sigma, odom_sigma) -> None:
        super().__init__(intrinsics, prior_sigma, odom_sigma)

    def _add_landmark(self, instance, add_noise=False):
        for obj_id, bbox in zip(instance.object_key, instance.bbox):
            sum_wid_hight = bbox[2:].sum() - bbox[:2].sum()
            bbox_sigma = [sum_wid_hight] * 4
            bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(bbox_sigma, dtype=float))
            if add_noise:
                bbox = np.random.multivariate_normal(bbox, bbox_noise.covariance())
            box = gtquadric.AlignedBox2(*bbox)
            bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(instance.image_key), L(obj_id), bbox_noise)
            self.graph.add(bbf)


class IncrementalSLAM(SLAM):
    def __init__(self, intrinsics, prior_sigma, odom_sigma, bbox_sigma=[20] * 4) -> None:
        super().__init__(intrinsics, prior_sigma, odom_sigma, bbox_sigma=bbox_sigma)
        self.std_quadric = None
        self.isam = self.optimizer()
        self.local_estimate = gtsam.Values()

    def make_graph(self, instances: Instances, add_landmarks=True, add_odom_noise=True, add_meas_noise=True):
        raise AttributeError

    def optimizer(self):
        # create isam optimizer 
        opt_params = gtsam.ISAM2DoglegParams()
        # opt_params.setVerbose(True)
        # opt_params.setWildfireThreshold(1e-08)
        params = gtsam.ISAM2Params()
        params.setOptimizationParams(opt_params)
        params.setEnableRelinearization(True)
        params.setRelinearizeThreshold(0.01)
        params.setRelinearizeSkip(1)
        params.setCacheLinearizedFactors(False)
        params.setFactorization('QR')
        # params.setEnablePartialRelinearizationCheck(True)
        # params.setEnableDetailedResults(True)
        isam = gtsam.ISAM2(params)

        return isam

    def solve(self, instances, add_odom_noise):
        # create storage for traj/map estimates
        current_trajectory = {}
        current_quadrics = {}

        # store quadrics until they have been viewed enough times to be constrained (>3)
        unconstrained_quadrics = {}

        wh_quadric = {q: [] for q in instances.bbox_ids}
        # Compute sum of width and height for each quadric using ALL bboxes
        for sample in instances:
            w_h_sum = (sample.bbox[:, 2:] - sample.bbox[:, :2]).sum(axis=1)
            for key, wh in zip(sample.object_key, w_h_sum):
                wh_quadric[key].append(wh)

        self.std_quadric = {q: gtsam.noiseModel.Diagonal.Sigmas(np.array([np.std(wh_quadric[q])] * 4, dtype=float))
                            for q in wh_quadric.keys()}

        step = 1
        counter = 0
        p_counter = 0

        gt_values = instances.toValues()

        # Params to filter bbox close to image boundaries
        image_bounds = gtquadric.AlignedBox2(0, 0, 640, 480)
        filter_pixels = 15
        filter_bounds = image_bounds.vector() + np.array([1, 1, -1, -1]) * filter_pixels
        filter_bounds = gtquadric.AlignedBox2(filter_bounds)

        for i, instance in enumerate(instances):
            if i == 0:
                curr_key = instance.image_key
                curr_pose = instance.pose
                prior_factor = gtsam.PriorFactorPose3(X(curr_key), curr_pose, self.prior_noise)
                self.local_estimate.insert(X(curr_key), curr_pose)
                self.graph.add(prior_factor)

            # else:
            #     curr_pose = instance.pose
            #     curr_key = instance.image_key
            #     previous_pose = instances[prev_key].pose
            #     if add_odom_noise:
            #         relative_pose_true = previous_pose.between(curr_pose)
            #         relative_rpy_xyz = np.hstack((relative_pose_true.rotation().rpy(),
            #                                       relative_pose_true.translation()))
            #         odom = np.random.multivariate_normal(relative_rpy_xyz,
            #                                              self.odometry_noise.covariance())
            #         odom = gtsam.Pose3(gtsam.Rot3.RzRyRx(odom[:3]), odom[3:6].reshape(-1, 1))
            #         curr_pose = previous_pose.compose(odom)
            #
            #     odom = previous_pose.between(curr_pose)
            #
            #     # compound odometry to global pose
            #     previous_pose = current_trajectory[prev_key]
            #     curr_pose = previous_pose.compose(odom)  # current pose in world frame
            #
            #     # add pose estimate to values and current estimateo (for initialization)
            #     self.local_estimate.insert(X(curr_key), curr_pose)
            #
            #     # add odometry factor to graph
            #     odom_factor = gtsam.BetweenFactorPose3(X(prev_key), X(curr_key), odom, self.odometry_noise)
            #     self.graph.add(odom_factor)

            else:
                curr_pose = instance.pose
                curr_key = instance.image_key
                previous_pose = instances[prev_key].pose
                # previous_pose = current_trajectory[prev_key]
                # print("Before noise")
                # print(curr_pose.rotation().rpy())
                # print(curr_pose.translation())
                if add_odom_noise:
                    noise = np.random.multivariate_normal(np.zeros(6), self.odometry_noise.covariance())
                    relative_pose_true = previous_pose.between(curr_pose)
                    curr_pose = previous_pose.compose(relative_pose_true.compose(relative_pose_true.Expmap(noise)))
                    # print("After noise")
                    # print(curr_pose.rotation().rpy())
                    # print(curr_pose.translation())
                odom = previous_pose.between(curr_pose)

                # compound odometry to global pose
                previous_pose = current_trajectory[prev_key]
                curr_pose = previous_pose.compose(odom)  # current pose in world frame
                # print('Weird change')
                # print(curr_pose.rotation().rpy())
                # print(curr_pose.translation())

                # add pose estimate to values and current estimateo (for initialization)
                self.local_estimate.insert(X(curr_key), curr_pose)

                # add odometry factor to graph
                odom_factor = gtsam.BetweenFactorPose3(X(prev_key), X(curr_key), odom, self.odometry_noise)
                self.graph.add(odom_factor)

                # print("add factor betweeen {} and {}".format(prev_key, curr_key))

            prev_key = curr_key

            self.cam_ids.append(curr_key)

            current_trajectory[curr_key] = curr_pose

            boxes = instance.bbox  # bbox of current frame
            # associate boxes -> quadrics 
            associated_keys = instance.object_key
            # print(associated_keys)
            # wrap boxes with keys 
            associated_boxes = []
            for box, quadric_key in zip(boxes, associated_keys):
                # if (quadric_key == 13 or quadric_key == 2 or quadric_key == 4) and counter >= 0 and filter_bounds.contains(gtquadric.AlignedBox2(*box)):
                if counter >= 0 and filter_bounds.contains(gtquadric.AlignedBox2(*box)):
                    counter = 0
                    associated_boxes.append({
                        'box': box,
                        'quadric_key': quadric_key,
                        'pose_key': curr_key,
                    })
            counter += 1
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
                if len(quadric_measurements) > 5:
                    # quadric = initialize_quadric(quadric_measurements, current_trajectory, self.calibration)
                    quadric = gtquadric.ConstrainedDualQuadric.getFromValues(gt_values, L(quadric_key))
                    quadric.addToValues(self.local_estimate, L(quadric_key))
                    current_quadrics[quadric_key] = quadric
                    for measurement in quadric_measurements:
                        box = measurement['box']
                        box = gtquadric.AlignedBox2(*box)
                        quadric_key = measurement['quadric_key']
                        pose_key = measurement['pose_key']
                        # bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                        #                                   self.bbox_noise, "STANDARD")
                        bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                                                          self.std_quadric[quadric_key], "STANDARD")
                        self.graph.add(bbf)
                        # Add Prior factor to landmarks seen during pose X(0)
                        # if pose_key == 0:
                        #     point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
                        #     q_prior = gtsam.PriorFactorPoint3(L(quadric_key), quadric.centroid(), point_noise)
                        #     self.graph.add(q_prior)
                    temp_dir.pop(quadric_key)

            unconstrained_quadrics = temp_dir
            # add measurements to graph if quadric is initialized and constrained
            current_quadrics_keys = current_quadrics.keys()
            for detection in old_boxes:
                box = detection['box']
                box = gtquadric.AlignedBox2(*box)
                quadric_key = detection['quadric_key']
                pose_key = detection['pose_key']
                bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                                                  self.std_quadric[quadric_key], "STANDARD")
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
                    drawing.box_and_text(gtquadric.AlignedBox2(*frame['box']), (0, 0, 255), text, (0, 0, 0))

                # visualize current view into map
                camera_pose = current_trajectory[max(current_trajectory.keys())]
                for quadric in current_quadrics.values():
                    drawing.quadric(camera_pose, quadric, self.calibration, (255, 0, 255))
                cv2.imshow('current view', image)
                # cv2.waitKey(0)

            self.isam.update(self.graph, self.local_estimate)
            self.isam.update()
            estimate = self.isam.calculateEstimate()
            # self.graph.saveGraph('current.dot', estimate)
            # self.isam.saveGraph('test.dot') # This is a clique graph
            # self.isam.marginalCovariance(X(1)) # Get covariance
            # self.isam.getFactorsUnsafe()
            # print(self.isam.getDelta())


            # Report all current state estimates from the iSAM2 optimization.
            if p_counter >= 20:
                p_counter = 0
                report_on_progress(self.isam, estimate, curr_key, pose_each=20)
            p_counter += 1
            # clear graph/estimate
            self.graph.resize(0)
            self.local_estimate.clear()

            # update the estimated quadrics and trajectory
            for j in range(len(estimate.keys())):
                key = estimate.keys()[j]
                if chr(gtsam.symbolChr(key)) == 'l':
                    quadric = gtquadric.ConstrainedDualQuadric.getFromValues(estimate, key)
                    current_quadrics[gtsam.symbolIndex(key)] = quadric
                elif chr(gtsam.symbolChr(key)) == 'x':
                    current_trajectory[gtsam.symbolIndex(key)] = estimate.atPose3(key)

            step += 1
            print(step)
            print(self.evaluate(gt_values, estimate))

        return estimate

    def evaluate(self, gt, results):
        gt_cam = []
        esti_cam = []
        for i in range(len(results.keys())):
            key = results.keys()[i]
            if chr(gtsam.symbolChr(key)) == 'x':
                gt_cam.append(gt.atPose3(key).translation())
                esti_cam.append(results.atPose3(key).translation())
        cam_rmse = np.array(
            [np.linalg.norm(gt_pose - esti_pose) for gt_pose, esti_pose in zip(gt_cam, esti_cam)]).mean()

        return {"Cam RMSE": cam_rmse}


import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot


def report_on_progress(graph: gtsam.ISAM2, current_estimate: gtsam.Values,
                       key: int, pose_each: int = 10):
    """Print and plot incremental progress of the robot for 2D Pose SLAM using iSAM2."""

    # Print the current estimates computed using iSAM2.
    # print("*"*50 + f"\nInference after State {key+1}:\n")
    # print(current_estimate)

    # Compute the marginals for all states in the graph.
    # marginals = gtsam.Marginals(graph, current_estimate)

    # Plot the newly updated iSAM2 inference.
    fig = plt.figure(0)
    axes = fig.gca(projection='3d')
    plt.cla()

    # update the estimated quadrics and trajectory
    pose_counter = 0
    for j in range(len(current_estimate.keys())):
        key = current_estimate.keys()[j]
        if chr(gtsam.symbolChr(key)) == 'l':
            quadric = gtquadric.ConstrainedDualQuadric.getFromValues(current_estimate, key)
            # This covariance is huge!
            # gtsam_plot.plot_pose3(0, quadric.pose(), 0.35,
            #                       graph.marginalCovariance(key))
            gtsam_plot.plot_pose3(0, quadric.pose(), 0.1)
            # print('Added new Quadric!!!')
            evals = np.linalg.eigvals(graph.marginalCovariance(key))
            print("Max {} and Min {} Eigen values".format(np.max(evals), np.min(evals)))
            # print(evals)
            # print(np.linalg.det(graph.marginalCovariance(key)))
        elif chr(gtsam.symbolChr(key)) == 'x':
            if pose_counter % pose_each == 0:
                gtsam_plot.plot_pose3(0, current_estimate.atPose3(key), 0.5,
                                      graph.marginalCovariance(key))
                # gtsam_plot.plot_pose3(0, current_estimate.atPose3(key), 0.35)
                pose_counter = 1
                # print('Added new pose!!!')
                # print(graph.marginalCovariance(key))
            pose_counter += 1

    axes.set_xlim3d(-2, 3)
    axes.set_ylim3d(-2, 3)
    axes.set_zlim3d(0, 2)
    plt.pause(0.1)
