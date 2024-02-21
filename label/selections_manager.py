import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d
from open3d import visualization

from label.selections import Selection, Selections
from label.utils.area_gt import AreaGT
from label.utils.commons import crop_cloud, load_cloud
from label.utils.tools import pick_points, pick_w_pinhole, print_title
from label.utils.user import user_input, user_question


class SelectionsManager:
    SIZE_DS = 0.008
    CROP_AREA = 0.35
    CROP_HEIGHT = 0.3

    def __init__(
        self,
        path_cloud,
        T_raw,
        path_save,
        name,
        params: dict,
        load_confirmation=True,
    ) -> None:
        self.params = params
        self.load_confirmation = load_confirmation
        self.T = np.array(T_raw)
        self.clear = lambda: os.system("clear")
        self.name = name
        self.pinhole = None
        self.display_real_color = True
        self.initMessage(self.name)

        # Loading saved progress
        self.path_selections = Path(path_save)
        self.selections: Selections = Selections(self.T)
        self.loadSelections()

        # Creating GT area
        area_gt = AreaGT()
        area_gt.fromPoints(self.params.gt.GT_MIN, self.params.gt.GT_MAX)

        # Initializing clouds
        self.cloud = load_cloud(path_cloud, self.T)
        self.cloud = area_gt.cropCloud(self.cloud)
        self.cloud_tree = o3d.geometry.KDTreeFlann(self.cloud)
        self.cloud_ds = self.cloud.voxel_down_sample(self.SIZE_DS)
        self.cloud_ds_tree = o3d.geometry.KDTreeFlann(self.cloud_ds)
        self.cloud_ds_color = o3d.geometry.PointCloud()
        self.cloud_crop = o3d.geometry.PointCloud()
        self.cloud_crop_tree = o3d.geometry.KDTreeFlann()

        # Initialize base parameters
        self.view_pose = np.eye(4)
        self.ask_selection = True
        self.inferColors()

    def initMessage(self, name):
        self.clear()
        print_title("Fruit selection: {}".format(name), "-")

    def saveSelections(self):
        # Create folder
        self.path_selections.mkdir(exist_ok=True)
        path_ply = self.path_selections / "ply"
        path_ply.mkdir(exist_ok=True)
        path_position = self.path_selections / "position"
        path_position.mkdir(exist_ok=True)
        path_radius = self.path_selections / "radius"
        path_radius.mkdir(exist_ok=True)

        # Save json
        path_json = self.path_selections / "data.json"
        file = open(path_json, "w")
        json_str = self.selections.toJson()
        file.write(json_str)
        file.close()

        center_radius = self.selections.get_center_radius()
        cloud_points = np.asarray(self.cloud.points)
        cloud_colors = np.asarray(self.cloud.colors)

        for idx, (center, radius) in center_radius.items():
            _, idxs, _ = self.cloud_tree.search_radius_vector_3d(center, radius)
            points = cloud_points[idxs]
            colors = cloud_colors[idxs]
            # Create a point cloud
            point_cloud = o3d.geometry.PointCloud()

            # Set the points and colors
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Save ply
            path_cloud = path_ply / (str(idx) + ".ply")
            o3d.io.write_point_cloud(path_cloud.as_posix(), point_cloud)

            # Save position
            path_position_data = path_position / (str(idx) + ".txt")
            file = open(path_position_data, "w")
            file.write(str(center.tolist()))
            file.close()

            # Save radius
            path_radius_data = path_radius / (str(idx) + ".txt")
            file = open(path_radius_data, "w")
            file.write(str(radius))
            file.close()

    def inferColors(self):
        print(" - infering colors")

        cloud_tree = o3d.geometry.KDTreeFlann(self.cloud)
        selections_list = self.selections.getList()
        for sel in selections_list:
            mean, std = get_fruit_color_std(
                self.cloud, cloud_tree, sel.center, sel.radius
            )
            if std == [0, 0, 0]:
                self.selections.removeClose(sel.center)
                continue
            sel.color_mean = mean
            sel.color_std = std

        print(" - infering done")
        self.saveSelections()

    def loadSelections(self):
        path_data = self.path_selections / "data.json"
        if not os.path.exists(path_data):
            return

        file = open(path_data, "r")
        selections = Selections(self.T).fromJson(file.read())
        file.close()

        if len(selections) == 0:
            return

        if not self.load_confirmation:
            print(" - loaded {} fruits selections".format(len(selections)))
            self.selections = selections
            return

        if user_question(
            " - found {} fruits selections, continue from there?".format(
                len(selections)
            )
        ):
            self.selections = selections

    def userSelection(self):
        self.initMessage(self.name)

        if not self.ask_selection:
            return True

        print(
            "Fruit selection:",
            " -> Continue",
            " 1) Remove selection",
            " 2) Change view point",
            " 3) Change display color ({})".format(
                "real" if self.display_real_color else "random"
            ),
            " 4) Exit",
            sep=os.linesep,
        )
        action = user_input("")

        if action == None:
            if not self.isViewPointSet():
                self.selectViewPoint()
        elif action == 1:
            if not self.isViewPointSet():
                self.selectViewPoint()
            self.removeSelection()
        elif action == 2:
            self.selectViewPoint()
        elif action == 3:
            self.changeDisplayColor()
            self.updateCloudDS()
            self.updateCloudCrop()
        elif action == 4:
            return False

        return True

    def changeDisplayColor(self):
        if self.display_real_color:
            self.display_real_color = False
        else:
            self.display_real_color = True

    def isViewPointSet(self):
        if (self.view_pose == np.eye(4)).all():
            return False
        return True

    def userSuperSelection(self):
        while True:
            self.updateCloudDS()
            self.updateCloudCrop()
            user_continue = self.userSelection()

            if not user_continue:
                # Exit
                return

            self.selectFruit()
            self.saveSelections()

    def updateCloudDS(self, ids_color: Dict[int, List[float]] = {}):
        self.cloud_ds_color = deepcopy(self.cloud_ds)
        color_cloud(
            self.cloud_ds_color,
            self.cloud_ds_tree,
            self.selections.getList(),
            ids_color,
            real_color=self.display_real_color,
        )

    def updateCloudCrop(self, ids_connections_color: Dict[int, List[float]] = {}):
        if not self.isViewPointSet():
            return

        cloud_copy = deepcopy(self.cloud)
        cloud_copy.transform(np.linalg.inv(self.view_pose))

        self.cloud_crop = crop_cloud(
            cloud_copy, size_area=self.CROP_AREA, size_height=self.CROP_HEIGHT
        )
        self.cloud_crop.transform(self.view_pose)
        self.cloud_crop_tree = o3d.geometry.KDTreeFlann(self.cloud_crop)

        color_cloud(
            self.cloud_crop,
            self.cloud_crop_tree,
            self.selections.getList(),
            ids_connections_color,
            self.display_real_color,
        )

    def selectViewPoint(self):
        self.updateCloudDS()

        while True:
            idxs = pick_points(self.cloud_ds_color, "View point selection")
            if len(idxs) == 0:
                self.ask_selection = True
                return

            if len(idxs) == 1:
                break

            print("Attention: select only one point")

        position = self.cloud_ds.points[idxs[0]]
        self.view_pose = np.eye(4)
        self.view_pose[:-1, -1] = position

        self.updateCloudCrop()
        self.ask_selection = False
        self.pinhole = None

    def updateViewPoint(self, view_pose: np.ndarray):
        self.view_pose = view_pose
        self.pinhole = None

    def selectFruit(self):
        # Pick points
        idxs, self.pinhole = pick_w_pinhole(self.cloud_crop, self.pinhole)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) < 4:
            print("Attention: select at least four points")
            self.ask_selection = True
            return

        self.addFruit(idxs)

        sel = self.selections.getList()[-1]
        color_cloud(
            self.cloud_crop,
            self.cloud_crop_tree,
            [sel],
            real_color=self.display_real_color,
        )
        self.ask_selection = False

    def addFruit(self, idxs):
        points = self.getFruitPoints(idxs)
        center, radius = fit_sphere(points)
        color, std = get_fruit_color_std(
            self.cloud_crop, self.cloud_crop_tree, center, radius
        )
        self.selections.add(center, radius, color, std)

    def getFruitPoints(self, idxs):
        points = []
        for idx in idxs:
            points.append(self.cloud_crop.points[idx])
        points = np.array(points)
        return points

    def removeSelection(self):
        print(
            "Remove selection:", " 1) Pick", " 2) Last", " 3) Go back", sep=os.linesep
        )
        action = user_input("")

        if action == 1:
            self.removePickSelection()
        elif action == 2:
            self.removeLastSelection()
        else:
            return

    def removeLastSelection(self):
        if len(self.selections) == 0:
            print("Attention: no selection available to be removed")
            return

        self.selections.removeLast()

        self.updateCloudCrop()

    def removePickSelection(self):
        # Pick points
        idxs, self.pinhole = pick_w_pinhole(self.cloud_crop, self.pinhole)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 1:
            print("Attention: select only one point")
            self.ask_selection = True
            return

        point = self.cloud_crop.points[idxs[0]]
        self.selections.removeClose(point)

        self.updateCloudCrop()
        self.ask_selection = False

    def getFromPosition(self, position) -> Selection:
        return self.selections.getClosest(position, dist_max=0.1)


def fit_sphere(points: np.ndarray):
    # credits to: https://jekel.me/2015/Least-Squares-Sphere-Fit/

    A = np.ones((points.shape[0], points.shape[1] + 1))
    A[:, :-1] = points * 2

    #   Assemble the f matrix
    f = np.zeros((points.shape[0], 1))
    f[:, 0] = (
        (points[:, 0] * points[:, 0])
        + (points[:, 1] * points[:, 1])
        + (points[:, 2] * points[:, 2])
    )
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)

    C = C.flatten()
    center = np.array([C[0], C[1], C[2]])

    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return center, radius


COLOR_SALT = [0.9, 0.9, 0.9]
COLOR_PEPPER = [0.6, 0.6, 0.6]
COLOR_BLUE = [0, 0, 1]
COLOR_PURPLE = [0.498, 0, 1]
COLORS = [COLOR_BLUE, COLOR_PURPLE]


def color_cloud(
    cloud: o3d.geometry.PointCloud,
    cloud_tree: o3d.geometry.KDTreeFlann,
    selections: List[Selection],
    ids_conns_color: Dict[int, List[float]] = {},
    real_color=False,
):
    for sel in selections:
        _, idxs, _ = cloud_tree.search_radius_vector_3d(sel.center, sel.radius)

        color_fruit = sel.color_mean if real_color else sel.color_display
        color_unique = random.choice(COLORS)

        fruit_colors = np.asarray(cloud.colors)
        for idx in idxs:
            if sel.id in ids_conns_color:
                color = ids_conns_color[sel.id]
                color = add_noise(color)
            elif ids_conns_color:
                color = color_unique
            else:
                color = color_fruit
                color = add_noise(color)

            fruit_colors[idx] = color


def add_noise(color):
    if random.uniform(0, 1) < 1 / 8:
        color = COLOR_SALT
    elif random.uniform(0, 1) < 1 / 4:
        color = COLOR_PEPPER

    return color


def get_fruit_color_std(
    cloud: o3d.geometry.PointCloud,
    cloud_tree: o3d.geometry.KDTreeFlann,
    center: np.ndarray,
    radius: float,
):
    colors = []
    for radius in np.arange(radius * 0.8, radius * 1, 0.05):
        _, idxs, _ = cloud_tree.search_radius_vector_3d(center, radius)
        for idx in idxs:
            colors.append(cloud.colors[idx])
    colors = np.array(colors)
    try:
        color = list(colors.mean(axis=0))
        std = list(colors.std(axis=0))
    except TypeError:
        return [1, 0, 0], [0, 0, 0]
    return color, std
