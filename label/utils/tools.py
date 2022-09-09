import cv2
import random
import webcolors
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from typing import List
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from label.types.pose import Pose
from scipy.spatial import KDTree

EPSILON = 0.000003


def print_title(msg, char="="):
    msg = " " + msg + " "
    lenght = max(60, len(msg))
    print("\n" + char * lenght)
    print(msg)
    print(char * lenght)


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def display_image(image, block=True):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    plt.axis("off")
    plt.imshow(image)
    plt.show(block=block)
    plt.pause(0.000000000001)


def display_two_images(image1, image2, block=True):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    fig = plt.figure(figsize=(20, 10))

    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("08")
    plt.axis("off")

    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("14")
    plt.axis("off")

    plt.show(block=block)
    #  plt.pause(0.000000000001)


def interpolation(p_prev, p_next, ts_inter):
    R_prev_aft = R.from_matrix([p_next.getRotation(), p_prev.getRotation()])

    dt = (ts_inter - p_prev.timestamp) / (p_next.timestamp - p_prev.timestamp)
    position = p_prev.getPosition() + dt * (p_next.getPosition() - p_prev.getPosition())
    slerp = Slerp([0, 1], R_prev_aft)
    rotation = slerp(dt)

    return Pose(rotation=rotation.as_matrix(), position=position)


def interpolation_ds(p_prev: Pose, p_next: Pose, ds: float):
    R_prev_aft = R.from_matrix([p_next.getRotation(), p_prev.getRotation()])

    position = p_prev.getPosition() + ds * (p_next.getPosition() - p_prev.getPosition())
    slerp = Slerp([0, 1], R_prev_aft)
    rotation = slerp(ds)

    return Pose(rotation=rotation.as_matrix(), position=position)


def image_to_world(image, intrinsic, extrinsic, depth) -> np.ndarray:
    # Image to Camera frame
    X_image_ = image * depth
    X_camera = np.linalg.pinv(intrinsic) @ X_image_

    # Camera to World coordinate system
    X_camera = np.array([X_camera[2], -X_camera[0], -X_camera[1], 1])

    # Camera to World frame
    X_world = extrinsic @ X_camera

    return X_world


def world_to_image(X_world, intrinsic, extrinsic, homogeneous=False) -> np.ndarray:
    if X_world.shape[0] == 3:
        X_world = np.append(X_world, 1)

    # World to Camera frame
    X_camera = np.linalg.pinv(extrinsic) @ X_world

    # World to Camera coordinate system
    X_camera = np.array([-X_camera[1], -X_camera[2], X_camera[0], 1])

    # Camera to Image frame
    image = intrinsic @ X_camera

    if homogeneous:
        return image

    depth = image[2]
    image /= depth
    image = np.round(image).astype(int)

    return image


def compute_norm(position1, position2):
    norm = np.linalg.norm(position1 - position2)
    return norm


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def convert_rgb_to_names(rgb_uniform):
    rgb = np.array(rgb_uniform) * 255

    # a dictionary of all the hex and their respective names in css3
    names: List[str] = []
    rgb_values = []
    for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb)
    return names[index]


def depth_from_disparity(
    x_cur: float, x_prev: float, baseline: float, focal: float
) -> float:
    disparity = x_cur - x_prev
    depth = baseline * focal / (disparity + 0.0001)
    # Clip to prevent crazy values
    depth = np.clip(depth, 0.05, 1.5)

    return depth


def reject_outliers(data, m=1.0):
    if data.shape[0] < 3:
        return data

    abs_v = abs(data - np.mean(data, axis=0))
    std_v = np.std(data, axis=0)
    data_new = data[(abs_v < m * std_v).any(axis=1)]

    if data_new.shape[0] < 1:
        return data

    return data_new


def sample_circle(radius):
    theta = random.uniform(0, np.pi)
    phi = random.uniform(0, 2 * np.pi)

    r = radius * 0.98
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.array([x, y, z])


def get_color_unique():
    color_blue = [0, 0, 1]
    color_purple = [0.498, 0, 1]

    if random.uniform(0, 1) < 1 / 2:
        color_unique = color_blue
    else:
        color_unique = color_purple

    return color_unique


def get_noisy_color(color):
    color_salt = [0.8, 0.8, 0.8]

    if random.uniform(0, 1) < 1 / 4:
        color = color
    else:
        color = color_salt

    return color


def ICP(source, target, voxel_size=0.05, num_iterations=50000):
    print(" - ransac ...")
    source_ds = source.voxel_down_sample(voxel_size)
    target_ds = target.voxel_down_sample(voxel_size)
    src_fpfh = compute_features(source_ds, voxel_size)
    trg_fpfh = compute_features(target_ds, voxel_size)
    ransac_result = execute_global_registration(
        source_ds, target_ds, src_fpfh, trg_fpfh, voxel_size
    )
    T_ransac = ransac_result.transformation

    print(" - refine ...")
    T_icp = refine_registration(
        source_ds, target_ds, T_ransac, voxel_size, num_iterations
    )

    return T_icp


def compute_features(cloud_ds, voxel_size):
    radius_normal = voxel_size * 2
    cloud_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    cloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return cloud_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(30000, 0.999),
    )
    return result


def refine_registration(
    source, target, initial_guess, voxel_size, num_iterations=50000
):
    radius_normal = voxel_size
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    distance_threshold = voxel_size
    kernel_threshold = 0.3 * voxel_size
    robust_kernel = o3d.pipelines.registration.GMLoss(kernel_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(robust_kernel),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            1e-06, 1e-06, int(num_iterations)
        ),
    )
    return result.transformation
