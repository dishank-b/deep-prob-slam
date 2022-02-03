import numpy as np
import cv2

import gtsam
import gtsam_quadrics as gtquadric
from gtsam.symbol_shorthand import X, L

from instances import Instances
from quadrics_multiview import initialize_quadric
from drawing import CV2Drawing
from visualization import report_on_progress


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
        self.global_graph = gtsam.NonlinearFactorGraph()
        self.global_values = gtsam.Values()

    def make_graph(self, instances: Instances, add_landmarks=True, add_odom_noise=True, add_meas_noise=True):
        raise AttributeError

    def optimizer(self):
        # create isam optimizer 
        opt_params = gtsam.ISAM2DoglegParams()
        opt_params.setAdaptationMode('ONE_STEP_PER_ITERATION')
        # opt_params = gtsam.ISAM2GaussNewtonParams()
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
        # self.std_quadric = {q: gtsam.noiseModel.Diagonal.Sigmas(np.array([40] * 4, dtype=float))
        #                     for q in wh_quadric.keys()}

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
            if i % 1 != 0:
                continue
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
                if add_odom_noise:
                    noise = np.random.multivariate_normal(np.zeros(6), self.odometry_noise.covariance())
                    relative_pose_true = previous_pose.between(curr_pose)
                    curr_pose = previous_pose.compose(relative_pose_true.compose(relative_pose_true.Expmap(noise)))
                odom = previous_pose.between(curr_pose)

                # compound odometry to global pose
                previous_pose = current_trajectory[prev_key]
                curr_pose = previous_pose.compose(odom)  # current pose in world frame

                # add pose estimate to values and current estimateo (for initialization)
                self.local_estimate.insert(X(curr_key), curr_pose)

                # add odometry factor to graph
                odom_factor = gtsam.BetweenFactorPose3(X(prev_key), X(curr_key), odom, self.odometry_noise)
                self.graph.add(odom_factor)

                # Add prior over previous poses
                if curr_key < 2:
                    prior_factor = gtsam.PriorFactorPose3(X(curr_key), instance.pose, self.prior_noise)
                    self.graph.add(prior_factor)

                # print("add factor betweeen {} and {}".format(prev_key, curr_key))

            prev_key = curr_key

            self.cam_ids.append(curr_key)

            current_trajectory[curr_key] = curr_pose

            boxes = instance.bbox  # bbox of current frame
            covs = instance.bbox_covar
            # associate boxes -> quadrics 
            associated_keys = instance.object_key
            # print(associated_keys)
            # wrap boxes with keys
            associated_boxes = []
            for box, cov, quadric_key in zip(boxes, covs, associated_keys):
                if (quadric_key != 37 and quadric_key != 41 and quadric_key != 40) and counter >= 0 and filter_bounds.contains(gtquadric.AlignedBox2(*box)):
                # if counter >= 0 and filter_bounds.contains(gtquadric.AlignedBox2(*box)):
                    counter = 0
                    associated_boxes.append({
                        'box': box,
                        'cov': cov,
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
                if len(quadric_measurements) > 30:
                    quadric = gtquadric.ConstrainedDualQuadric.getFromValues(gt_values, L(quadric_key))
                    quadric.addToValues(self.local_estimate, L(quadric_key))
                    current_quadrics[quadric_key] = quadric
                    for measurement in quadric_measurements:
                        box = measurement['box']
                        bbox_covar = measurement['cov']
                        box = gtquadric.AlignedBox2(*box)
                        quadric_key = measurement['quadric_key']
                        pose_key = measurement['pose_key']
                        # bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                        #                                   self.std_quadric[quadric_key], "STANDARD")
                        bbox_noise = gtsam.noiseModel.Gaussian.Covariance(np.array(bbox_covar, dtype=float))
                        bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                                                          bbox_noise, "STANDARD")
                        self.graph.add(bbf)
                        # Add Prior factor to landmarks seen during pose X(0)
                        angle_factor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([100] * 3, dtype=np.float))
                        q_factor = gtquadric.QuadricAngleFactor(L(quadric_key), gtquadric.ConstrainedDualQuadric().pose().rotation(),
                                                                angle_factor_noise)
                        self.graph.add(q_factor)
                    temp_dir.pop(quadric_key)

            unconstrained_quadrics = temp_dir
            # add measurements to graph if quadric is initialized and constrained
            current_quadrics_keys = current_quadrics.keys()
            for detection in old_boxes:
                box = detection['box']
                bbox_covar = measurement['cov']
                bbox_noise = gtsam.noiseModel.Gaussian.Covariance(np.array(bbox_covar, dtype=float))
                box = gtquadric.AlignedBox2(*box)
                quadric_key = detection['quadric_key']
                pose_key = detection['pose_key']
                # bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                #                                   self.std_quadric[quadric_key], "STANDARD")
                bbf = gtquadric.BoundingBoxFactor(box, self.calibration, X(pose_key), L(quadric_key),
                                                  bbox_noise, "STANDARD")
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
                    # camera_pose = instance.pose
                    for key, quadric in current_quadrics.items():
                        if key == frame['quadric_key']:
                            drawing.quadric(camera_pose, quadric, self.calibration, (255, 0, 255))
                # cv2.imshow('current view', image)
                # cv2.waitKey(0)

            # add to global graph/values
            self.global_graph.push_back(self.graph)
            self.global_values.insert(self.local_estimate)

            # print('\nchecking global graph/values')
            # v1 = self.valid_system(self.global_graph, self.global_values)

            # Preventing Undetermined error
            try:
                self.isam.update(self.graph, self.local_estimate)
                estimate = self.isam.calculateEstimate()
                # self.graph.saveGraph('current.dot', estimate)
                # self.isam.saveGraph('test.dot') # This is a clique graph
                # self.isam.marginalCovariance(X(1)) # Get covariance
                # self.isam.getFactorsUnsafe()
                # print(self.isam.getDelta())
            except Exception as e:
                print("Not optimized at step {}".format(step))
                # clear graph/estimate
                self.graph.resize(0)
                self.local_estimate.clear()
                step += 1
                print(e)
                continue

            # Report all current state estimates from the iSAM2 optimization.
            if p_counter >= 20:
                p_counter = 0
                report_on_progress(self.isam, estimate, pose_each=20, pose_cov=False, quadric_cov=False)
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

    def valid_system(self, nlfg, values):
        '''
        https://github.com/best-of-acrv/gtsam-quadrics/blob/7faa25d4cb7ce19a2a37304df047e2eee466bb40/quadricslam/quadricslam_online.py
        '''

        gfg = nlfg.linearize(values)
        jacobian = gfg.jacobian()[0]
        hessian = gfg.hessian()[0]
        valid = True

        # check if underdetermined
        if np.linalg.matrix_rank(jacobian) < values.dim() \
                or np.linalg.matrix_rank(hessian) < values.dim():
            print('NOT VALID: Undetermined system')
            valid = False

        # check if indefinite, i.e not positive semidefinite or negative semidefinite
        eigv = np.linalg.eigh(hessian)[0]
        if np.any(eigv < 0) and np.any(eigv > 0):
            print('NOT VALID: indefinite hessian')
            valid = False

        if not np.all(eigv > 0):
            print('NOT VALID: not postive definite')
            valid = False

        # Check conditioning
        cond = np.linalg.cond(jacobian)
        print('  Conditioning: ', cond)

        # COND CHECKING:
        # check almost underconstrained variable
        # vastly different uncertainties
        return valid



