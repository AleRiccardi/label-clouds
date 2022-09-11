from contextlib import redirect_stdout
import os
import io
import math
import random
import numpy as np
import open3d as o3d

from copy import deepcopy
from typing import Dict, List

from label.utils.area_gt import AreaGT
from label.selections import Selections, Selection
from label.utils.commons import pick_points, crop_cloud, load_cloud
from label.utils.parameters import Parameters
from label.utils.tools import get_screen_size, print_title
from label.utils.user import user_input, user_question


class SelectionsManager:
    SIZE_DS = 0.01
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
        self.initMessage(self.name)

        # Loading saved progress
        self.path_selections = path_save
        self.selections: Selections = Selections(self.T)
        self.loadSelections()

        # Creating GT area
        area_gt = AreaGT()
        area_gt.fromPoints(self.params.gt.GT_MIN, self.params.gt.GT_MAX)

        # Initializing clouds
        self.cloud = load_cloud(path_cloud, self.T)
        self.cloud = area_gt.cropCloud(self.cloud)
        self.cloud_ds = self.cloud.voxel_down_sample(self.SIZE_DS)
        self.cloud_ds_tree = o3d.geometry.KDTreeFlann(self.cloud_ds)
        self.cloud_ds_color = o3d.geometry.PointCloud()
        self.cloud_crop = o3d.geometry.PointCloud()
        self.cloud_crop_tree = o3d.geometry.KDTreeFlann()

        # Initialize base parameters
        self.view_pose = np.eye(4)
        self.ask_selection = True

    def initMessage(self, name):
        self.clear()
        print_title("Fruit selection: {}".format(name), "-")

    def saveSelections(self):
        file = open(self.path_selections, "w")
        json_str = self.selections.toJson()
        file.write(json_str)
        file.close()

    def loadSelections(self):
        if not os.path.exists(self.path_selections):
            return

        file = open(self.path_selections, "r")
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
            " 3) Exit",
            sep=os.linesep,
        )
        action = user_input("")

        if action == None:
            if (self.view_pose == np.eye(4)).all():
                self.selectViewPoint()
        elif action == 1:
            self.removeSelection()
        elif action == 2:
            self.selectViewPoint()
        elif action == 3:
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
            self.cloud_ds_color, self.cloud_ds_tree, self.selections.get(), ids_color
        )

    def updateCloudCrop(self, ids_color: Dict[int, List[float]] = {}):
        cloud_copy = deepcopy(self.cloud)
        cloud_copy.transform(np.linalg.inv(self.view_pose))

        self.cloud_crop = crop_cloud(
            cloud_copy, size_area=self.CROP_AREA, size_height=self.CROP_HEIGHT
        )
        self.cloud_crop.transform(self.view_pose)
        self.cloud_crop_tree = o3d.geometry.KDTreeFlann(self.cloud_crop)

        color_cloud(
            self.cloud_crop, self.cloud_crop_tree, self.selections.get(), ids_color
        )

    def selectViewPoint(self):
        self.updateCloudDS()

        while True:
            idxs = self.pick_view(self.cloud_ds_color)

            if len(idxs) != 1:
                print("Attention: select one point")
                continue

            position = self.cloud_ds.points[idxs[0]]
            self.view_pose = np.eye(4)
            self.view_pose[:-1, -1] = position

            self.updateCloudCrop()
            self.ask_selection = False
            break

    def selectFruit(self):
        # Pick points
        idxs = self.pick_fruits(self.cloud_crop)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) < 4:
            print("Attention: select at least four points")
            self.ask_selection = True
            return

        points = []
        for idx in idxs:
            points.append(self.cloud_crop.points[idx])
        points = np.array(points)

        center, radius = fit_sphere(points)
        self.selections.add(center, radius * 1.2)

        sel = self.selections.get()[-1]
        color_cloud(self.cloud_crop, self.cloud_crop_tree, [sel])
        self.ask_selection = False

    def pick_view(self, cloud):
        vis = o3d.visualization.VisualizerWithEditing()
        w, h = get_screen_size()
        vis.create_window(
            window_name="View point selection",
            width=w,
            height=h,
            left=0,
            top=0,
            visible=True,
        )
        vis.add_geometry(cloud)

        f = io.StringIO()
        with redirect_stdout(f):
            # user picks points
            vis.run()

        vis.destroy_window()
        return vis.get_picked_points()

    def pick_fruits(self, cloud):
        vis = o3d.visualization.VisualizerWithEditing()
        w, h = get_screen_size()
        vis.create_window(
            window_name="View point selection",
            width=w,
            height=h,
            left=0,
            top=0,
            visible=True,
        )
        vis.add_geometry(cloud)
        view = vis.get_view_control()

        f = io.StringIO()
        with redirect_stdout(f):
            # user picks points
            vis.run()

        vis.destroy_window()
        self.fov = view.get_field_of_view()
        print(self.fov)
        return vis.get_picked_points()

    def removeSelection(self):
        print(
            "Remove selection:", " 1) Last", " 2) Pick", " 3) Go back", sep=os.linesep
        )
        action = user_input("")

        if action == 1:
            self.removeLastSelection()
        elif action == 2:
            self.removePickSelection()
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
        idxs = pick_points(self.cloud_crop)
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


def color_cloud(
    cloud: o3d.geometry.PointCloud,
    cloud_tree: o3d.geometry.KDTreeFlann,
    selections: List[Selection],
    ids_color: Dict[int, List[float]] = {},
):
    color_salt = [0.8, 0.8, 0.8]
    color_blue = [0, 0, 1]
    color_purple = [0.498, 0, 1]

    for sel in selections:
        _, idxs, _ = cloud_tree.search_radius_vector_3d(sel.center, sel.radius)

        color = sel.color
        if random.uniform(0, 1) < 1 / 2:
            color_unic = color_blue
        else:
            color_unic = color_purple

        fruit_colors = np.asarray(cloud.colors)
        for idx in idxs:
            if ids_color:
                if sel.id in ids_color:
                    color = ids_color[sel.id]
                elif random.uniform(0, 1) < 1 / 4:
                    color = color_salt
                else:
                    color = color_unic

            fruit_colors[idx] = color
