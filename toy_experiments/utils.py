"""Various plotting utlities."""

# pylint: disable=no-member, invalid-name

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import gtsam


def set_axes_equal(fignum):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      fignum (int): An integer representing the figure number for Matplotlib.
    """
    fig = plt.figure(fignum)
    ax = fig.gca(projection='3d')

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def ellipsoid(rx, ry, rz, n):
    """
    Numpy equivalent of Matlab's ellipsoid function.

    Args:
        rx (double): Radius of ellipsoid in X-axis.
        ry (double): Radius of ellipsoid in Y-axis.
        rz (double): Radius of ellipsoid in Z-axis.
        n (int): The granularity of the ellipsoid plotted.

    Returns:
        tuple[numpy.ndarray]: The points in the x, y and z axes to use for the surface plot.
    """
    u = np.linspace(0, 2*np.pi, n+1)
    v = np.linspace(0, np.pi, n+1)
    x = -rx * np.outer(np.cos(u), np.sin(v)).T
    y = -ry * np.outer(np.sin(u), np.sin(v)).T
    z = -rz * np.outer(np.ones_like(u), np.cos(v)).T

    return x, y, z


def plot_covariance_ellipse_3d(axes, origin, P, scale=1, n=8, alpha=0.5):
    """
    Plots a Gaussian as an uncertainty ellipse

    Based on Maybeck Vol 1, page 366
    k=2.296 corresponds to 1 std, 68.26% of all probability
    k=11.82 corresponds to 3 std, 99.74% of all probability

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        origin (gtsam.Point3): The origin in the world frame.
        P (numpy.ndarray): The marginal covariance matrix of the 3D point
            which will be represented as an ellipse.
        scale (float): Scaling factor of the radii of the covariance ellipse.
        n (int): Defines the granularity of the ellipse. Higher values indicate finer ellipses.
        alpha (float): Transparency value for the plotted surface in the range [0, 1].
    """
    k = 11.82
    U, S, _ = np.linalg.svd(P)

    radii = k * np.sqrt(S)
    radii = radii * scale
    rx, ry, rz = radii

    # generate data for "unrotated" ellipsoid
    xc, yc, zc = ellipsoid(rx, ry, rz, n)

    # rotate data with orientation matrix U and center c
    data = np.kron(U[:, 0:1], xc) + np.kron(U[:, 1:2], yc) + \
        np.kron(U[:, 2:3], zc)
    n = data.shape[1]
    x = data[0:n, :] + origin[0]
    y = data[n:2*n, :] + origin[1]
    z = data[2*n:, :] + origin[2]

    axes.plot_surface(x, y, z, alpha=alpha, cmap='hot')


def plot_pose2_on_axes(axes, pose, axis_length=0.1, covariance=None):
    """
    Plot a 2D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        pose (gtsam.Pose2): The pose to be plotted.
        axis_length (float): The length of the camera axes.
        covariance (numpy.ndarray): Marginal covariance matrix to plot
            the uncertainty of the estimation.
    """
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    t = pose.translation()
    origin = t

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'g-')

    if covariance is not None:
        pPp = covariance[0:2, 0:2]
        gPp = np.matmul(np.matmul(gRp, pPp), gRp.T)

        w, v = np.linalg.eig(gPp)

        # k = 2.296  # corresponding - 1 std
        k = 5.0 # corresponding to 2 std?

        angle = np.arctan2(v[1, 0], v[0, 0])
        e1 = patches.Ellipse(origin, np.sqrt(w[0]*k), np.sqrt(w[1]*k),
                             np.rad2deg(angle), fill=False)
        axes.add_patch(e1)


def plot_pose2(fignum, pose, axis_length=0.1, covariance=None,
               axis_labels=('X axis', 'Y axis', 'Z axis')):
    """
    Plot a 2D pose on given figure with given `axis_length`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        pose (gtsam.Pose2): The pose to be plotted.
        axis_length (float): The length of the camera axes.
        covariance (numpy.ndarray): Marginal covariance matrix to plot
            the uncertainty of the estimation.
        axis_labels (iterable[string]): List of axis labels to set.
    """
    # get figure object
    fig = plt.figure(fignum)
    axes = fig.gca()
    plot_pose2_on_axes(axes, pose, axis_length=axis_length,
                       covariance=covariance)

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])

    return fig


def plot_point3_on_axes(axes, point, linespec, P=None):
    """
    Plot a 3D point on given axis `axes` with given `linespec`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
    """
    axes.plot([point[0]], [point[1]], [point[2]], linespec)
    if P is not None:
        plot_covariance_ellipse_3d(axes, point, P)


def plot_point3(fignum, point, linespec, P=None,
                axis_labels=('X axis', 'Y axis', 'Z axis')):
    """
    Plot a 3D point on given figure with given `linespec`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
        axis_labels (iterable[string]): List of axis labels to set.

    Returns:
        fig: The matplotlib figure.

    """
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plot_point3_on_axes(axes, point, linespec, P)

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])

    return fig


def plot_3d_points(fignum, values, linespec="g*", marginals=None,
                   title="3D Points", axis_labels=('X axis', 'Y axis', 'Z axis')):
    """
    Plots the Point3s in `values`, with optional covariances.
    Finds all the Point3 objects in the given Values object and plots them.
    If a Marginals object is given, this function will also plot marginal
    covariance ellipses for each point.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        values (gtsam.Values): Values dictionary consisting of points to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        marginals (numpy.ndarray): Marginal covariance matrix to plot the
            uncertainty of the estimation.
        title (string): The title of the plot.
        axis_labels (iterable[string]): List of axis labels to set.
    """

    keys = values.keys()

    # Plot points and covariance matrices
    for key in keys:
        try:
            point = values.atPoint3(key)
            if marginals is not None:
                covariance = marginals.marginalCovariance(key)
            else:
                covariance = None

            fig = plot_point3(fignum, point, linespec, covariance,
                              axis_labels=axis_labels)

        except RuntimeError:
            continue
            # I guess it's not a Point3

    fig = plt.figure(fignum)
    fig.suptitle(title)
    fig.canvas.set_window_title(title.lower())


def plot_pose3_on_axes(axes, pose, axis_length=0.1, P=None, scale=1):
    """
    Plot a 3D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
    """
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    origin = pose.translation()

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')

    # plot the covariance
    if P is not None:
        # covariance matrix in pose coordinate frame
        pPp = P[3:6, 3:6]
        # convert the covariance matrix to global coordinate frame
        gPp = gRp @ pPp @ gRp.T
        plot_covariance_ellipse_3d(axes, origin, gPp)


def plot_pose3(fignum, pose, axis_length=0.1, P=None,
               axis_labels=('X axis', 'Y axis', 'Z axis')):
    """
    Plot a 3D pose on given figure with given `axis_length`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        pose (gtsam.Pose3): 3D pose to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
        axis_labels (iterable[string]): List of axis labels to set.

    Returns:
        fig: The matplotlib figure.
    """
    # get figure object
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plot_pose3_on_axes(axes, pose, P=P,
                       axis_length=axis_length)

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])

    return fig


def plot_trajectory(fignum, values, scale=0.5, marginals=None,
                    title="Plot Trajectory", axis_labels=('X axis', 'Y axis', 'Z axis'), **kwargs):
    """
    Plot a complete 2D/3D trajectory using poses in `values`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        values (gtsam.Values): Values containing some Pose2 and/or Pose3 values.
        scale (float): Value to scale the poses by.
        marginals (gtsam.Marginals): Marginalized probability values of the estimation.
            Used to plot uncertainty bounds.
        title (string): The title of the plot.
        axis_labels (iterable[string]): List of axis labels to set.
    """
    fig = plt.figure(fignum)
    axes = fig.gca()

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])

    # Plot 2D poses, if any
    poses = gtsam.utilities.allPose2s(values)
    for key in poses.keys():
        pose = poses.atPose2(key)
        if marginals:
            covariance = marginals.marginalCovariance(key)
        else:
            covariance = None

        plot_pose2_on_axes(axes, pose, covariance=covariance,
                           axis_length=scale)
    axes.plot(*zip(*[poses.atPose2(key).translation() for key in poses.keys()]), **kwargs)
    axes.axis('equal')
    axes.legend()
    fig.suptitle(title)
    fig.canvas.set_window_title(title.lower())


def plot_incremental_trajectory(fignum, values, start=0,
                                scale=1, marginals=None,
                                time_interval=0.0):
    """
    Incrementally plot a complete 3D trajectory using poses in `values`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        values (gtsam.Values): Values dict containing the poses.
        start (int): Starting index to start plotting from.
        scale (float): Value to scale the poses by.
        marginals (gtsam.Marginals): Marginalized probability values of the estimation.
            Used to plot uncertainty bounds.
        time_interval (float): Time in seconds to pause between each rendering.
            Used to create animation effect.
    """
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    poses = gtsam.utilities.allPose3s(values)
    keys = gtsam.KeyVector(poses.keys())

    for key in keys[start:]:
        if values.exists(key):
            pose_i = values.atPose3(key)
            plot_pose3(fignum, pose_i, scale)

    # Update the plot space to encompass all plotted points
    axes.autoscale()

    # Set the 3 axes equal
    set_axes_equal(fignum)

    # Pause for a fixed amount of seconds
    plt.pause(time_interval)

def plot_landmarks(fignum, point2s, landmark_keys, marginals=None, **kwargs):
    fig = plt.figure(fignum)
    axes = fig.gca()

    axes.scatter(*zip(*[point2s.atPoint2(key) for key in landmark_keys]), **kwargs)

    nstd = 2

    if marginals:
        for key in landmark_keys:
            covar = marginals.marginalCovariance(key)
            eigvals, eigvecs = np.linalg.eigh(covar)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            # The anti-clockwise angle to rotate our ellipse by 
            vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
            theta = np.arctan2(vy, vx)

            # Width and height of ellipse to draw
            width, height = 2 * nstd * np.sqrt(eigvals)

            ellipse = patches.Ellipse(xy=point2s.atPoint2(key), width=width, height=height,
                        angle=np.degrees(theta), fill = None)
            
            axes.add_patch(ellipse)
    
    axes.legend()
    axes.axis('equal')



