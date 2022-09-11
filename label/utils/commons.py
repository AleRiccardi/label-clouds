import os
import io
import numpy as np
import open3d as o3d

from contextlib import redirect_stdout


def set_view_pose(vis: o3d.visualization.VisualizerWithEditing, view_pose):
    view_control = vis.get_view_control()
    camera_parameters = view_control.convert_to_pinhole_camera_parameters()
    camera_parameters.extrinsic = view_pose
    view_control.convert_from_pinhole_camera_parameters(camera_parameters)


def crop_cloud(
    cloud: o3d.geometry.PointCloud,
    size_area: float = 3,
    size_height: float = 2,
    display_msg=False,
):
    if display_msg:
        print("Cropping the cloud ...")
    poligon = get_points_cuboid(size_area, size_height=size_height)
    points = o3d.utility.Vector3dVector(poligon)
    bnb = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    return cloud.crop(bnb)


def get_points_cuboid(size_x: float = 3, size_y: float = 2, size_height: float = 2):
    center = {"x": 0, "y": 0, "z": 0}
    poligon = np.array(
        [
            # Vertices Polygon1
            [
                center["x"] + (size_x / 2),
                center["y"] + size_y,
                center["z"] + size_height,
            ],  # face-topright
            [
                center["x"] - (size_x / 2),
                center["y"] + size_y,
                center["z"] + size_height,
            ],  # face-topleft
            [
                center["x"] - (size_x / 2),
                center["y"] - size_y,
                center["z"] + size_height,
            ],  # rear-topleft
            [
                center["x"] + (size_x / 2),
                center["y"] - size_y,
                center["z"] + size_height,
            ],  # rear-topright
            # Vertices Polygon 2
            [
                center["x"] + (size_x / 2),
                center["y"] + size_y,
                center["z"] - size_height,
            ],
            [
                center["x"] - (size_x / 2),
                center["y"] + size_y,
                center["z"] - size_height,
            ],
            [
                center["x"] - (size_x / 2),
                center["y"] - size_y,
                center["z"] - size_height,
            ],
            [
                center["x"] + (size_x / 2),
                center["y"] - size_y,
                center["z"] - size_height,
            ],
        ]
    ).astype("float64")
    return poligon


def load_cloud(path_cloud, T=np.eye(4)):
    _, name = os.path.split(path_cloud)
    print(' - loading "{}"'.format(name))
    cloud = o3d.io.read_point_cloud(path_cloud)
    cloud.transform(T)
    print(" - loading done")

    return cloud


def add_clouds(cloud1, cloud2):
    points = []
    points.extend(np.array(cloud1.points))
    points.extend(np.array(cloud2.points))
    colors = []
    colors.extend(np.array(cloud1.colors))
    colors.extend(np.array(cloud2.colors))

    return create_cloud(points, colors)


def create_cloud(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def interpolation(poses, ts_prev, ts_next, ts_inter):
    p_prev = poses[ts_prev]
    p_next = poses[ts_next]
    R_prev_aft = R.from_matrix([p_prev[:, :-1], p_next[:, :-1]])

    dt = (ts_inter - ts_prev) / (ts_next - ts_prev)
    trans = p_prev[:, -1] + dt * (p_next[:, -1] - p_prev[:, -1])
    slerp = Slerp([0, 1], R_prev_aft)
    rotation = slerp(dt)

    pose = np.eye(4)
    pose[:-1, :-1] = rotation.as_matrix()
    pose[:-1, -1] = trans
