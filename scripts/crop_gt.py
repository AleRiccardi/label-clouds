import click
import numpy as np
import open3d as o3d


def crop_cloud(
    cloud: o3d.geometry.PointCloud,
    center: dict,
    size_area: float = 3,
    size_height: float = 2,
):
    points = o3d.utility.Vector3dVector(
        get_points_cuboid(center, size_area, size_height)
    )
    bnb = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    return cloud.crop(bnb)


def get_points_cuboid(center, size_area: float = 3, size_height: float = 2):
    return np.array(
        [
            # Vertices Polygon1
            [
                center["x"] + (size_area / 2),
                center["y"] + (size_area / 2),
                center["z"] + size_height,
            ],  # face-topright
            [
                center["x"] - (size_area / 2),
                center["y"] + (size_area / 2),
                center["z"] + size_height,
            ],  # face-topleft
            [
                center["x"] - (size_area / 2),
                center["y"] - (size_area / 2),
                center["z"] + size_height,
            ],  # rear-topleft
            [
                center["x"] + (size_area / 2),
                center["y"] - (size_area / 2),
                center["z"] + size_height,
            ],  # rear-topright
            # Vertices Polygon 2
            [
                center["x"] + (size_area / 2),
                center["y"] + (size_area / 2),
                center["z"] - size_height,
            ],
            [
                center["x"] - (size_area / 2),
                center["y"] + (size_area / 2),
                center["z"] - size_height,
            ],
            [
                center["x"] - (size_area / 2),
                center["y"] - (size_area / 2),
                center["z"] - size_height,
            ],
            [
                center["x"] + (size_area / 2),
                center["y"] - (size_area / 2),
                center["z"] - size_height,
            ],
        ]
    ).astype("float64")


@click.command()
@click.argument("path_cloud")
@click.option("--path_tr")
def main(path_cloud, path_tr):
    T_20_14 = np.loadtxt(path_tr)

    print("Reading the point cloud")
    cloud = o3d.io.read_point_cloud(path_cloud)
    cloud = cloud.transform(T_20_14)

    start = np.array([27.5, 0.6, 0.5])
    stop = np.array([33.5, 0.6, 0.5])
    center = (stop + start) / 2
    center = {"x": center[0], "y": center[1], "z": center[2]}

    print("Cropping the point cloud")
    cloud_crop = crop_cloud(cloud, center)
    o3d.visualization.draw_geometries([cloud_crop])


if __name__ == "__main__":
    main()
