"""
QuadricSLAM Copyright 2020, ARC Centre of Excellence for Robotic Vision, Queensland University of Technology (QUT)
Brisbane, QLD 4000
All Rights Reserved

See LICENSE for the license information

Description: Drawing interface
Author: Lachlan Nicholson (Python)
"""

# import standard libraries
import os
import sys
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, CheckButtons

# import gtsam and extension
import gtsam
from gtsam.symbol_shorthand import X, L
# L = lambda i: int(gtsam.symbol(ord('l'), i))
# X = lambda i: int(gtsam.symbol(ord('x'), i))

import gtsam_quadrics as gtquadric
from evaluation import Evaluation
import visualization

# modify system path so file will work when run directly or as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 16})
rcParams['axes.labelpad'] = 20
sys.dont_write_bytecode = True

# BAD_LANDMARKS = []
# Indeterminate system error landmarks:
# Large radii (np.linalg.norm(quadric.radii()) > 0.7)
#BAD_LANDMARKS = [58, 1, 2, 26, 21, 47, 56, 51, 48, 54, 0]
# Low detections (detections > 20)
BAD_LANDMARKS = [48, 54, 47, 56, 58, 177, 112, 50, 43, 49, 30, 32, 27, 39, 8,
        41, 55, 9, 19, 51, 18, 37, 16, 0]
# Dog Leg indeterminate system
#BAD_LANDMARKS = [10, 12, 30, 47, 48, 54, 56, 58 ]
#BAD_LANDMARKS = [2, 3, 4, 8, 10, 12, 14, 19, 17, 23, 24, 30, 32, 38, 43, 47, 48, 50, 51, 54, 56, 58,
#        112, 177]

#BAD_LANDMARKS = [8, 38, 48, 58]
#BAD_LANDMARKS = [0, 14, 23]

class IterationVisualizer(object):

    def __init__(self) -> None:
        self.iteration_values = {}
        self.iteration_index = 0;
        self.gt_values = None
        self.graph = None
        fig = plt.figure()
        self.ax = fig.gca(projection='3d')
        self.bbox_ids = []

    def add_bbox_ids(self, bbox_ids) -> None:
        self.bbox_ids = bbox_ids

    def add_iteration(self, values) -> None:
        self.iteration_values[self.iteration_index] = values
        self.iteration_index += 1

    def add_gt_trajectory(self, gt_values) -> None:
        self.gt_values = gt_values

    def add_graph(self, graph) -> None:
        self.graph = graph

    def plot(self) -> None:
        # Make a horizontel oriented slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label="interation",
            valmin=0,
            valmax=self.iteration_index-1,
            valstep=1,
            valinit=0,
            orientation="horizontal"
        )
        self.slider.on_changed(self.__update_plot)

        rax = plt.axes([0.05, 0.7, 0.15, 0.15])
        self.check = CheckButtons(rax, ('Trajectories', 'Quadrics', 'Errors'))
        self.check.on_clicked(self.__update_plot)


        plt.show()

    def __update_plot(self, val) -> None:

        self.ax.clear()
        status = self.check.get_status()
        values = self.iteration_values[self.slider.val]
        if status[0]:
            R = np.eye(3)
            t = np.ones((3,1))
            poses = gtsam.utilities.allPose3s(values)

            est_t = np.array([poses.atPose3(k).translation() for k in poses.keys()]).T
            if self.gt_values is not None:
                gt_poses = gtsam.utilities.allPose3s(self.gt_values)
                gt_t = np.array([gt_poses.atPose3(k).translation() for k in
                    gt_poses.keys()]).T
                R, t, _ = Evaluation._horn_align(np.matrix(est_t), np.matrix(gt_t))
                self.ax.plot(gt_t[0,:], gt_t[1,:], gt_t[2, :], color='g', label="Ground Truth")



            est_t = np.array(np.hstack([ R *
                np.matrix(poses.atPose3(k).translation()).T + t for k
                in poses.keys()]))
            self.ax.plot(est_t[0,:], est_t[1,:], est_t[2, :], color='b', label="Estimated")

        if status[1]:
            marginals = gtsam.Marginals(self.graph, values)
            det_marginal_cov = []
            for id in self.bbox_ids:
                if id in BAD_LANDMARKS:
                    continue
                quadric = gtquadric.ConstrainedDualQuadric.getFromValues(values,
                        L(id))
                self.__plot_quadric(quadric)
                #det_marginal_cov[np.linalg.norm(quadric.radii())] = id
                det_marginal_cov.append((np.linalg.det(marginals.marginalCovariance(L(id))),
                    id))

            print(dict(sorted(det_marginal_cov, key=lambda item: item[0])))

        if status[2]:
            error_to_factor = {}
            for i in range(0,self.graph.size()):
                factor = self.graph.at(i)
                error_to_factor[factor.error(values)] = factor

            sorted_dict = dict(sorted(error_to_factor.items(), key=lambda item: item[0]))
            sorted_dict = dict(list(sorted_dict.items())[-100:])
            smallest_e = list(sorted_dict.items())[0][0]
            largest_e = list(sorted_dict.items())[-1][0]
            color_map = cm.ScalarMappable(colors.Normalize(smallest_e,
                largest_e), "YlOrRd")
            for error, factor in sorted_dict.items():
                points = np.empty((0,3))
                print(f"{factor.print()} error: {error}")
                print(f"Linearized: \n {factor.linearize(values).print()}")
                print(f"Jaconbian: \n {factor.linearize(values).jacobian()}")
                for k in factor.keys():
                    # 108 is for L (landmark) symbols
                    if gtsam.symbolChr(k) == 108:
                        quadric = gtquadric.ConstrainedDualQuadric.getFromValues(values, k)
                        points = np.vstack([points, np.array(quadric.centroid())])

                    # 120 is for X (pose) sybmols
                    elif gtsam.symbolChr(k) == 120:
                        points = np.vstack([points,
                            np.array(values.atPose3(k).translation())])
                self.ax.plot(points[:,0], points[:,1], points[:,2],
                        color=color_map.to_rgba(error), label="Errors")


        self.__set_axes_equal()

    def __plot_quadric(self, quadric) -> None:

        rotation = quadric.pose().rotation().matrix()
        translation = quadric.pose().translation()

        # Radii corresponding to the coefficients:
        rx, ry, rz = quadric.radii()

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)

        # Cartesian coordinates that correspond to the spherical angles:
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))
        # w = np.ones(x.shape)
        points = np.stack((x, y, z))

        points_2D = np.zeros((points.shape[1], points.shape[2], 2))

        # warp points to quadric (3D) and project to image (2D)
        for i in range(points.shape[1]):
            for j in range(points.shape[2]):
                # point = [x[i,j], y[i,j], z[i,j], 1.0]
                point = points[:, i, j]
                warped_point = point.dot(np.linalg.inv(rotation))
                warped_point += translation
                points[:, i, j] = warped_point

        self.ax.plot_wireframe(points[0,:,:], points[1,:, :], points[2, :, :],
                color='r')
    def __set_axes_equal(self):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        '''

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius]) 

class Visualizer(object):
    def __init__(self, cam_id, obj_id, calibration) -> None:
        super().__init__()
        self.fig = None
        self.cam_ids = cam_id
        self.bbox_ids = obj_id
        self.calibration = calibration
        # self.fig = plt.figure(figsize=(16, 12), dpi=100)
        self.reset_figure()
        self.eval = Evaluation(self.cam_ids)

    def reset_figure(self, dims=(10, 16)):
        self.fig = plt.figure(figsize=dims, dpi=100)

    def visualize(self, instances, values, gt_pose=False, save_video=False, video_name="video", box_ids = []):
        """
        Visualizing the value estimate of quadrics and trajectory
        """
        if save_video:
            video = cv2.VideoWriter(video_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (640, 480))
        
        colors = {0: (0, 255, 255), 1:(255, 0, 255), 2: (255, 255, 0)}
        obj_color = {obj_id: colors[np.random.randint(0, 3)] for obj_id in self.bbox_ids}
       
        for instance in instances:
            image = self._draw_instance(instance, values, gt_pose, box_ids, obj_color)
            if save_video:
                video.write(image)
            else:
                cv2.imshow("image", image)
                cv2.waitKey(0)
        
        if save_video:
            video.release()

                

    def _draw_instance(self, instance, values, gt_pose, box_ids, obj_color):
        image_path = instance.image_path
        image = cv2.imread(image_path)
        draw_gt = CV2Drawing(image)
        
        for obj_id, bbox, covar in zip(instance.object_key, instance.bbox, instance.bbox_covar):
            box = gtquadric.AlignedBox2(*bbox)
            quadric = gtquadric.ConstrainedDualQuadric.getFromValues(values, L(obj_id))
            if gt_pose:
                pose = instance.pose
            else:
                pose = values.atPose3(X(instance.image_key))
            draw_gt.quadric(pose, quadric, self.calibration, obj_color[obj_id])
            if obj_id in box_ids:
                draw_gt.box_and_text(box, (0, 0, 255), str(obj_id), (0, 0, 0), covar, (0, 255, 255))
            else:
                draw_gt.box_and_text(box, (255, 255, 0), str(obj_id), (0, 0, 0), covar, (0, 255, 255))
        return image


    def plot_comparison(self, gt, results, title="Comparison", add_landmarks=True,
                        labels=['Ground Truth', 'Estimated'], type='horn'):
        # Align Trajectories
        gt_cam, esti_cam = self.eval.align_trajectories(results, gt, type)

        # Plot Trajectory
        gt_cam = np.array([pose.translation() for pose in gt_cam]).T
        esti_cam = np.array([pose.translation() for pose in esti_cam]).T
        ax = self.add_plot('3d')
        trajectory = visualization.compare_3d(ax, gt_cam, esti_cam, title, labels=labels)

        # plots.update({"Cam Trajectory": trajectory})

        if add_landmarks:
            init_quad = np.array(
                [gtquadric.ConstrainedDualQuadric.getFromValues(gt, L(id)).centroid() for id in self.bbox_ids]).T
            esti_quad = np.array(
                [gtquadric.ConstrainedDualQuadric.getFromValues(results, L(id)).centroid() for id in self.bbox_ids]).T
            ax = self.add_plot('3d')
            quads = visualization.compare_quadrics(ax, init_quad, esti_quad, title, labels=labels)
            # plots.update({"Quadrics pose": quads})

        # return plots

    def plot_trajectory(self, values, title="Trajectory", add_landmarks=True, labels=[]):
        plots = {}
        n_plots = 2 if add_landmarks else 1
        cam_poses = np.array([values.atPose3(X(id)).translation() for id in self.cam_ids]).T
        ax = self.add_plot(n_plots=1)
        trajectory = visualization.visualize_trajectory(ax, cam_poses, title)
        plots.update({"Cam Trajectory": trajectory})

        if add_landmarks:
            quad_poses = np.array(
                [gtquadric.ConstrainedDualQuadric.getFromValues(values, L(id)).centroid() for id in self.bbox_ids]).T
            ax = self.add_plot(n_plots=n_plots)
            quad_poses = visualization.visualize_quadrics(ax, quad_poses, title)
            plots.update({"Quadrics pose": quad_poses})

        return plots

    def add_plot(self, projection=None, n_plots=2):
        n = len(self.fig.axes)
        # if n == 0:
        #     ax = self.fig.add_subplot(111, projection=projection)
        # else:
        #     for i in range(n):
        #         self.fig.axes[i].change_geometry(1, n + 1, i + 1)
        #
        ax = self.fig.add_subplot(n_plots, 1, n + 1, projection=projection)

        return ax


class CV2Drawing(object):
    def __init__(self, image):
        self._image = image
        self.image_width = self._image.shape[1]
        self.image_height = self._image.shape[0]

    def box(self, box, color=(0, 0, 255), thickness=2):
        cv2.rectangle(self._image, (int(box.xmin()), int(box.ymin())), (int(box.xmax()), int(box.ymax())), color,
                      thickness)

    def covar_ellipse(self, covar, box, color=(255, 0, 0), thickness=2):
        std = np.sqrt(covar)
        cv2.ellipse(self._image, (int(box.xmin()), int(box.ymin())), (int(4*std[0,0]), int(4*std[1,1])), 0, 0., 360, color, thickness)
        cv2.ellipse(self._image, (int(box.xmax()), int(box.ymax())), (int(4*std[2,2]), int(4*std[3,3])), 0, 0., 360, color, thickness)

    def text(self, text, lower_left, color=(255, 255, 255), thickness=1, background=False, background_color=(0, 0, 255),
             background_margin=3):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        lower_left = [lower_left[0] + background_margin, lower_left[1] - background_margin]
        text_size = cv2.getTextSize(text, font, font_scale, thickness)
        text_width = text_size[0][0] + background_margin * 2
        text_height = text_size[0][1] + background_margin * 2

        # lower_left = [upper_left[0], upper_left[1]+text_width]
        final_position = list(lower_left)
        final_position[1] = int(np.clip(lower_left[1], text_height, self.image_height))
        final_position[0] = int(np.clip(lower_left[0], 0, self.image_width - text_width))
        final_position = tuple(final_position)

        if (background):
            upper_left = [final_position[0], final_position[1] - text_height]
            xmin = upper_left[0] - background_margin + 1
            ymin = upper_left[1] - background_margin + 4
            xmax = upper_left[0] + text_width + background_margin
            ymax = upper_left[1] + text_height + background_margin + 1
            cv2.rectangle(self._image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), background_color, cv2.FILLED)
        cv2.putText(self._image, text, final_position, font, font_scale, color, thickness, cv2.LINE_AA)

    def box_and_text(self, box, box_color, text, text_color, covar, covar_color):
        box_thickness = 2
        # self.box(box, box_color, thickness=box_thickness)
        self.text(text, (box.xmin() - box_thickness, box.ymin() - box_thickness), text_color, 1, True, box_color)
        # self.covar_ellipse(covar, box, covar_color)

    def quadric(self, pose, quadric, calibration, color=(255, 0, 0), alpha=1):
        """ 
        Draws a wireframe quadric at camera position.
        Will not draw lines if both ends project outside image border.
        Will not draw if quadric is behind camera 
        """
        if quadric.isBehind(pose):
            return

        image_box = gtquadric.AlignedBox2(0, 0, self.image_width, self.image_height)
        points_2D = self.generate_uv_spherical(quadric, pose, calibration, 20, 20)
        points_2D = np.round(points_2D).astype('int')
        # color = (0,0,255)
        color = (color[2], color[1], color[0])  # rgb to bgr

        if alpha != 1:
            full_image = self._image.copy()

        for i in range(points_2D.shape[0]):
            for j in range(points_2D.shape[1] - 1):
                point_2D = points_2D[i, j]
                nextpoint_2D = points_2D[i, j + 1]
                if image_box.contains(gtsam.Point2(*point_2D)) or image_box.contains(gtsam.Point2(*nextpoint_2D)):
                    cv2.line(self._image, (point_2D[0], point_2D[1]), (nextpoint_2D[0], nextpoint_2D[1]), color, 1,
                             cv2.LINE_AA)

        for j in range(points_2D.shape[1]):
            for i in range(points_2D.shape[0] - 1):
                point_2D = points_2D[i, j]
                nextpoint_2D = points_2D[i + 1, j]
                if image_box.contains(gtsam.Point2(*point_2D)) or image_box.contains(gtsam.Point2(*nextpoint_2D)):
                    cv2.line(self._image, (point_2D[0], point_2D[1]), (nextpoint_2D[0], nextpoint_2D[1]), color, 1,
                             cv2.LINE_AA)

        if alpha != 1:
            cv2.addWeighted(self._image, alpha, full_image, 1 - alpha, 0, self._image)

    def generate_uv_spherical(self, quadric, pose, calibration, theta_points=30, phi_points=30):
        rotation = quadric.pose().rotation().matrix()
        translation = quadric.pose().translation()

        # Radii corresponding to the coefficients:
        rx, ry, rz = quadric.radii()

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, theta_points)
        v = np.linspace(0, np.pi, phi_points)

        # Cartesian coordinates that correspond to the spherical angles:
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))
        # w = np.ones(x.shape)
        points = np.stack((x, y, z))

        points_2D = np.zeros((points.shape[1], points.shape[2], 2))
        transform_to_image = gtquadric.QuadricCamera.transformToImage(pose, calibration)

        # warp points to quadric (3D) and project to image (2D)
        for i in range(points.shape[1]):
            for j in range(points.shape[2]):
                # point = [x[i,j], y[i,j], z[i,j], 1.0]
                point = points[:, i, j]
                warped_point = point.dot(np.linalg.inv(rotation))
                warped_point += translation

                point_3D = np.array([warped_point[0], warped_point[1], warped_point[2], 1.0])
                point_2D = transform_to_image.dot(point_3D)
                point_2D = point_2D[:-1] / point_2D[-1]
                points_2D[i, j] = point_2D

        return points_2D
