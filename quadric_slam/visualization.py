# Utitlity file with functions for handling trajectory plots
#
# Author: Miguel Saavedra

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd


def compare_3d(ax, ground_truth, trajectory, title, labels: List[str] = ['Ground Truth', 'Estimated']):
    """
    Plot the vehicle's trajectory in 3D space

    :param ground_truth: Numpy array (3 x M) where M is the number of samples
        with the ground truth trayectory of the vehicle.
    :param trajectory: Numpy array (3 x M) where M is the number of samples
        with the estimated trajectory of the vehicle.
    :param title: Name of the plot
    :param labels: Labels for the plot
    """
    # Axis limits
    maxX = np.amax(trajectory[0, :]) + 1.0
    minX = np.amin(trajectory[0, :]) - 1.0
    maxY = np.amax(trajectory[1, :]) + 1.0
    minY = np.amin(trajectory[1, :]) - 1.0
    maxZ = np.amax(trajectory[2, :]) + 1.0
    minZ = np.amin(trajectory[2, :]) - 1.0

    # est_traj_fig = plt.figure()
    # ax = est_traj_fig.add_subplot(111, projection='3d')
    ax.plot(ground_truth[0, :], ground_truth[1, :], ground_truth[2, :], "-", label=labels[0])
    ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], "-", label=labels[1])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title, y=1.15, fontweight="bold")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    ax.set_zlim(minZ, maxZ)
    plt.tight_layout()
    # ax.dist = 10
    # plt.show()

    # return est_traj_fig


def compare_quadrics(ax, gt, estimated, title="Quadrics poses", labels: List[str] = ['Ground Truth', 'Estimated']):
    """
    Plot the quadrics in 3D space

    :param gt: Numpy array (3 x M) where M is the number of quadrics
        with the ground truth 
    :param estimated: Numpy array (3 x M) where M is the number of quadrics
        with the estimated. 
    :param title: Name of the plot
    :param labels: Labels for the plot
    """

    # Axis limits
    maxX = np.amax(estimated[0, :]) + 1.0
    minX = np.amin(estimated[0, :]) - 1.0
    maxY = np.amax(estimated[1, :]) + 1.0
    minY = np.amin(estimated[1, :]) - 1.0
    maxZ = np.amax(estimated[2, :]) + 1.0
    minZ = np.amin(estimated[2, :]) - 1.0
    # est_traj_fig = plt.figure()
    # ax = est_traj_fig.add_subplot(111.0 projection='3d')
    ax.scatter(gt[0, :], gt[1, :], gt[2, :], c="blue", zorder=0, label=labels[0])
    ax.scatter(estimated[0, :], estimated[1, :], estimated[2, :], c="green", zorder=0, label=labels[1])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title, y=1.15, fontweight="bold")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    ax.set_zlim(minZ, maxZ)
    plt.tight_layout()
    # plt.show()

    # return est_traj_fig


def visualize_trajectory(trajectory, title="Vehicle's trajectory"):
    """
    Plot the vehicle's trajectory

    :param trajectory: Numpy array (3 x M) where M is the number of samples
        with the trayectory of the vehicle.
    :param title: Name of the plot
    """

    # lists for x, y and z values
    locX = list(trajectory[0, :])
    locY = list(trajectory[1, :])
    locZ = list(trajectory[2, :])

    # Axis limits
    maxX = np.amax(locX) + 1.0
    minX = np.amin(locX) - 1.0
    maxY = np.amax(locY) + 1.0
    minY = np.amin(locY) - 1.0
    maxZ = np.amax(locZ) + 1.0
    minZ = np.amin(locZ) - 1.0

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(2, 2)
    ZY_plt = plt.subplot(gspec[1, 1])
    YX_plt = plt.subplot(gspec[0, 1])
    ZX_plt = plt.subplot(gspec[1, 0])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    toffset = 1.0

    # Actual trajectory plotting ZX
    ZX_plt.set_title("Trajectory (X, Z)", y=toffset)
    ZX_plt.plot(locX, locZ, "--", label="Trajectory", zorder=1, linewidth=1.5, markersize=2)
    ZX_plt.set_xlabel("X [m]")
    ZX_plt.set_ylabel("Z [m]")
    # Plot vehicle initial location
    ZX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZX_plt.scatter(locX[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZX_plt.set_xlim([minX, maxX])
    ZX_plt.set_ylim([minZ, maxZ])
    ZX_plt.legend(bbox_to_anchor=(1.05, 1.0), loc=3, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    ZY_plt.set_title("Trajectory (Y, Z)", y=toffset)
    ZY_plt.set_xlabel("Y [m]")
    ZY_plt.plot(locY, locZ, "--", linewidth=1.5, markersize=2, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZY_plt.scatter(locY[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZY_plt.set_xlim([minY, maxY])
    ZY_plt.set_ylim([minZ, maxZ])

    # Plot YX
    YX_plt.set_title("Trajectory (Y X)", y=toffset)
    YX_plt.set_ylabel("X [m]")
    YX_plt.set_xlabel("Y [m]")
    YX_plt.plot(locY, locX, "--", linewidth=1.5, markersize=2, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    YX_plt.scatter(locY[-1], locX[-1], s=8, c="red", label="End location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([minX, maxX])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=1.1)
    D3_plt.plot3D(xs=locX, ys=locY, zs=locZ, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="green", zorder=1)
    D3_plt.scatter(locX[-1], locY[-1], locZ[-1], s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(minX, maxX)
    D3_plt.set_ylim3d(minY, maxY)
    D3_plt.set_zlim3d(minZ, maxZ)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X [m]", labelpad=0)
    D3_plt.set_ylabel("Y [m]", labelpad=0)
    D3_plt.set_zlabel("Z [m]", labelpad=-5)

    # Plotting the result
    fig.suptitle(title, fontsize=16, y=1.05)
    D3_plt.view_init(35, azim=45)
    plt.tight_layout()
    plt.show()

    return fig


def visualize_poses(trajectory, title="trajectory"):
    # locatoin = (3,M) where M is number of poses in trajectory with each pose = x,y,z
    locations = np.array([pose.translation() for pose in trajectory]).T

    # lists for x, y and z values
    locX = list(locations[0, :])
    locY = list(locations[1, :])
    locZ = list(locations[2, :])

    # Axis limits
    maxX = np.amax(locX) + 1
    minX = np.amin(locX) - 1
    maxY = np.amax(locY) + 1
    minY = np.amin(locY) - 1
    maxZ = np.amax(locZ) + 1
    minZ = np.amin(locZ) - 1

    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(16, 12), dpi=100)
    gspec = gridspec.GridSpec(1, 1)
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=1.1)
    D3_plt.plot3D(xs=locX, ys=locY, zs=locZ, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="green", zorder=1)
    D3_plt.scatter(locX[-1], locY[-1], locZ[-1], s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(minX, maxX)
    D3_plt.set_ylim3d(minY, maxY)
    D3_plt.set_zlim3d(minZ, maxZ)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X [m]", labelpad=0)
    D3_plt.set_ylabel("Y [m]", labelpad=0)
    D3_plt.set_zlabel("Z [m]", labelpad=-5)

    # plot orientation
    dx, dy, dz = np.eye(3)
    for pose in trajectory:
        R = pose.rotation().matrix()
        frame = ReferenceFrame(pose.translation(),
                               dx=R @ dx,
                               dy=R @ dy,
                               dz=R @ dz)
        frame.draw3d(ax=D3_plt)

        # Plotting the result
    fig.suptitle(title, fontsize=16, y=1.05)
    D3_plt.view_init(35, azim=45)
    plt.tight_layout()
    plt.show()


def visualize_quadrics(poses, title="Quadric poses"):
    """
    Plot the vehicle's trajectory

    :param trajectory: Numpy array (3 x M) where M is the number of samples
        with the trayectory of the vehicle.
    :param title: Name of the plot
    """

    # lists for x, y and z values
    locX = list(poses[0, :])
    locY = list(poses[1, :])
    locZ = list(poses[2, :])

    # Axis limits
    maxX = np.amax(locX) + 1
    minX = np.amin(locX) - 1
    maxY = np.amax(locY) + 1
    minY = np.amin(locY) - 1
    maxZ = np.amax(locZ) + 1
    minZ = np.amin(locZ) - 1

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(2, 2)
    ZY_plt = plt.subplot(gspec[1, 1])
    YX_plt = plt.subplot(gspec[0, 1])
    ZX_plt = plt.subplot(gspec[1, 0])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    toffset = 1.0

    # Actual trajectory plotting ZX
    ZX_plt.set_title("Trajectory (X, Z)", y=toffset)
    ZX_plt.scatter(locX, locZ, s=8, c="blue", label="Start location", zorder=2)
    ZX_plt.set_xlabel("X [m]")
    ZX_plt.set_ylabel("Z [m]")
    # Plot vehicle initial location
    ZX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZX_plt.scatter(locX[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZX_plt.set_xlim([minX, maxX])
    ZX_plt.set_ylim([minZ, maxZ])
    ZX_plt.legend(bbox_to_anchor=(1.05, 1.0), loc=3, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    ZY_plt.set_title("Trajectory (Y, Z)", y=toffset)
    ZY_plt.set_xlabel("Y [m]")
    ZY_plt.scatter(locY, locZ, s=8, c="blue", label="Start location", zorder=2)
    ZY_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZY_plt.scatter(locY[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZY_plt.set_xlim([minY, maxY])
    ZY_plt.set_ylim([minZ, maxZ])

    # Plot YX
    YX_plt.set_title("Trajectory (Y X)", y=toffset)
    YX_plt.set_ylabel("X [m]")
    YX_plt.set_xlabel("Y [m]")
    YX_plt.scatter(locY, locX, s=8, c="blue", label="Start location", zorder=2)
    YX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    YX_plt.scatter(locY[-1], locX[-1], s=8, c="red", label="End location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([minX, maxX])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=1.1)
    D3_plt.scatter(xs=locX, ys=locY, zs=locZ, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="green", zorder=1)
    D3_plt.scatter(locX[-1], locY[-1], locZ[-1], s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(minX, maxX)
    D3_plt.set_ylim3d(minY, maxY)
    D3_plt.set_zlim3d(minZ, maxZ)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X [m]", labelpad=0)
    D3_plt.set_ylabel("Y [m]", labelpad=0)
    D3_plt.set_zlabel("Z [m]", labelpad=-5)

    # Plotting the result
    fig.suptitle(title, fontsize=16, y=1.05)
    D3_plt.view_init(35, azim=45)
    plt.tight_layout()
    plt.show()

    return fig


class ReferenceFrame:
    def __init__(
            self,
            origin: np.ndarray,
            dx: np.ndarray,
            dy: np.ndarray,
            dz: np.ndarray
    ) -> None:
        self.origin = origin
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def draw3d(
            self,
            head_length: float = 0.1,
            color: list = ["red", "green", "blue"],
            ax: Optional[Axes3D] = None,
            name: str = ""
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca(projection="3d")

        # ax.text(*self.origin + 0.5, f"({name})")
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dx,
            head_length=head_length,
            color=color[0]
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dy,
            head_length=head_length,
            color=color[1]
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dz,
            head_length=head_length,
            color=color[2]
        )
        return ax


def draw3d_arrow(
        arrow_location: np.ndarray,
        arrow_vector: np.ndarray,
        head_length: float = 0.1,
        color: Optional[str] = None,
        name: Optional[str] = None,
        ax: Optional[Axes3D] = None,
) -> Axes3D:
    if ax is None:
        ax = plt.gca(projection="3d")

    ax.quiver(
        *arrow_location,
        *arrow_vector,
        length=head_length,
        arrow_length_ratio=head_length / np.linalg.norm(arrow_vector),
        color=color,
    )
    if name is not None:
        ax.text(*(arrow_location + arrow_vector), name)

    return ax
