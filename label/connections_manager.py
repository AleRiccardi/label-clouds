import os
import json
import socket
import random
import numpy as np
import open3d as o3d

from copy import deepcopy

from label.types.pose import Pose
from label.connections import Connections
from label.selections_manager import SelectionsManager
from label.selections import Selection
from label.utils.commons import add_clouds, pick_points
from label.utils.tools import print_title
from label.utils.user import user_input


class ConnectionsManager:
    TRANSFORM_14 = np.eye(4)
    TRANSFORM_14[:-1, -1] = [0, 1.2, 0]
    TRANSFORM_CLOSE_14 = np.eye(4)
    TRANSFORM_CLOSE_14[:-1, -1] = [0, 0.4, 0]
    HOST = "127.0.0.1"
    PORT = 9999

    def __init__(self, params) -> None:
        self.params = params
        self.view_pose = np.eye(4)
        self.clear = lambda: os.system("clear")
        self.initMessage()

        self.path_connections = params.path_connections
        path, _ = os.path.split(self.path_connections)

        if not os.path.exists(path):
            os.makedirs(path)

        self.connections: Connections = Connections()
        self.loadConnections()

        self.selection08 = SelectionsManager(
            params.path_cloud_1,
            params.transformations.gt_08,
            params.path_selections_1,
            "08",
            params,
            False,
        )
        self.selection14 = SelectionsManager(
            params.path_cloud_2,
            params.transformations.gt_14,
            params.path_selections_2,
            "14",
            params,
            False,
        )

        self.cloud_ds_08 = o3d.geometry.PointCloud()
        self.cloud_ds_14 = o3d.geometry.PointCloud()
        self.cloud_crop_08 = o3d.geometry.PointCloud()
        self.cloud_crop_14 = o3d.geometry.PointCloud()
        self.cloud_ds = o3d.geometry.PointCloud()
        self.cloud_crop = o3d.geometry.PointCloud()

        self.ask_selection = True
        #  self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #  self.socket.connect((self.HOST, self.PORT))

    def initMessage(self):
        self.clear()
        print_title("Fruit connections")

    def saveConnections(self):
        file = open(self.path_connections, "w")
        json_str = self.connections.toJson()
        file.write(json_str)
        file.close()

    def loadConnections(self):
        if not os.path.exists(self.path_connections):
            return

        file = open(self.path_connections, "r")
        json_type = json.loads(file.read())
        connections = Connections().fromJson(json_type)
        file.close()

        if len(connections) == 0:
            return

        print(" - found {} connections".format(len(connections)))
        self.connections = connections

    def userSelection(self):
        # Do not display selection if not asked by the user
        if not self.ask_selection:
            return True

        self.clear()
        print_title("Fruit connections")
        print(
            "\nFruit connection:",
            " -> Continue",
            " 1) Select a harvested fruit",
            " 2) Remove a connection",
            " 3) Select a view point",
            " 4) Enter the fruit selection",
            " 5) Exit",
            sep=os.linesep,
        )
        action = user_input("")

        if action == None:
            if (self.view_pose == np.eye(4)).all():
                self.selectViewPoint()
        elif action == 1:
            self.selectHarvestedFruit()
        elif action == 2:
            self.removeConnection()
        elif action == 3:
            self.selectViewPoint()
        elif action == 4:
            self.fruitSelection()
            self.selectViewPoint()
        elif action == 5:
            return False

        return True

    def removeConnection(self):
        print(
            "Remove connection:", " 1) Last", " 2) Pick", " 3) Go back", sep=os.linesep
        )
        action = user_input("")

        if action == 1:
            self.removeLastConnection()
        elif action == 2:
            self.removePickConnection()
        else:
            return

    def removeLastConnection(self):
        self.connections.removeLast()
        self.updateCloudCrop()
        self.ask_selection = True

    def removePickConnection(self):
        # Pick points
        idxs = pick_points(self.cloud_crop)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 1:
            print("Attention: select only one point")
            return

        sel08: Selection = Selection()
        sel14: Selection = Selection()
        for idx in idxs:

            if idx < len(self.cloud_crop_08.points):
                position = self.cloud_crop_08.points[idx]
                sel08 = self.selection08.getFromPosition(position)
            else:
                idx -= len(self.cloud_crop_08.points)
                position = self.cloud_crop_14.points[idx]
                pose = np.eye(4)
                pose[:-1, -1] = position
                pose = np.linalg.inv(self.TRANSFORM_CLOSE_14) @ pose
                position = pose[:-1, -1].flatten()
                sel14 = self.selection14.getFromPosition(position)

        if not sel08.isEmtpy():
            self.connections.remove(sel08.id, Connections.MAP_FIRST)
        elif not sel14.isEmtpy():
            self.connections.remove(sel14.id, Connections.MAP_SECOND)

        self.updateCloudCrop()
        self.ask_selection = True

    def fruitSelection(self):
        print("Chose:", " 1) Map 08", " 2) Map 14", " 3) Go back", sep=os.linesep)
        action = user_input("")

        if action == 1:
            self.selection08.userSuperSelection()
        elif action == 2:
            self.selection14.userSuperSelection()
        elif action == 3:
            return

    def updateCloudDS(self):
        ids_color_08 = self.connections.getIdsColor(Connections.MAP_FIRST)
        ids_color_14 = self.connections.getIdsColor(Connections.MAP_SECOND)

        self.selection08.updateCloudDS(ids_color_08)
        self.selection14.updateCloudDS(ids_color_14)

        self.cloud_ds_08 = self.selection08.cloud_ds_color
        self.cloud_ds_14 = deepcopy(self.selection14.cloud_ds_color).transform(
            self.TRANSFORM_14
        )

        self.cloud_ds = add_clouds(self.cloud_ds_08, self.cloud_ds_14)

    def updateCloudCrop(self):
        self.selection08.view_pose = self.view_pose
        self.selection14.view_pose = self.view_pose

        ids_color_08 = self.connections.getIdsColor(Connections.MAP_FIRST)
        ids_color_14 = self.connections.getIdsColor(Connections.MAP_SECOND)

        self.selection08.updateCloudCrop(ids_color_08)
        self.selection14.updateCloudCrop(ids_color_14)

        self.cloud_crop_08 = self.selection08.cloud_crop
        self.cloud_crop_14 = deepcopy(self.selection14.cloud_crop).transform(
            self.TRANSFORM_CLOSE_14
        )

        self.cloud_crop = add_clouds(self.cloud_crop_08, self.cloud_crop_14)

    def getCloudViewPoint(self, idx):
        if idx < len(self.cloud_ds_08.points):
            position = self.cloud_ds_08.points[idx]
            self.view_pose = np.eye(4)
            self.view_pose[:-1, -1] = position
        else:
            idx -= len(self.cloud_ds_08.points)
            position = self.cloud_ds_14.points[idx]
            self.view_pose = np.eye(4)
            self.view_pose[:-1, -1] = position
            self.view_pose = np.linalg.inv(self.TRANSFORM_14) @ self.view_pose

    def selectViewPoint(self):
        print("\n -> Select a view point")
        self.updateCloudDS()

        while True:
            idxs = pick_points(self.cloud_ds)
            if len(idxs) == 0:
                print("Attention: select one point")
                continue

            if len(idxs) == 1:
                break

            print("Attention: select only one point")

        self.getCloudViewPoint(idxs[0])
        self.updateCloudCrop()
        self.ask_selection = False
        #  self.sendSocketPose()

    def selectViewPointRandom(self):
        print("Selecting a random view point")
        self.updateCloudDS()
        idxs = [random.randint(0, len(self.cloud_ds.points))]
        self.getCloudViewPoint(idxs[0])
        self.updateCloudCrop()
        self.ask_selection = False
        #  self.sendSocketPose()

    def selectConnection(self):
        # Pick points
        idxs = pick_points(self.cloud_crop)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 2:
            print("Attention: select only two points")
            return

        sel08: Selection = Selection()
        sel14: Selection = Selection()
        for idx in idxs:

            if idx < len(self.cloud_crop_08.points):
                position = self.cloud_crop_08.points[idx]
                sel08 = self.selection08.getFromPosition(position)
            else:
                idx -= len(self.cloud_crop_08.points)
                position = self.cloud_crop_14.points[idx]
                pose = np.eye(4)
                pose[:-1, -1] = position
                pose = np.linalg.inv(self.TRANSFORM_CLOSE_14) @ pose
                position = pose[:-1, -1].flatten()
                sel14 = self.selection14.getFromPosition(position)

        if sel08.isEmtpy() or sel14.isEmtpy():
            print("Connection error")
            print(sel08.center)
            print(sel14.center)
            return

        self.connections.add(sel08.id, sel14.id)
        self.updateCloudCrop()
        self.ask_selection = False

    def selectHarvestedFruit(self):
        # Pick points
        idxs = pick_points(self.cloud_crop)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 1:
            print("Attention: select only one points")
            return

        sel08: Selection = Selection()
        for idx in idxs:
            if idx < len(self.cloud_crop_08.points):
                position = self.cloud_crop_08.points[idx]
                sel08 = self.selection08.getFromPosition(position)
            else:
                print("Error: removed fruit can be only in session 1")

        if sel08.isEmtpy():
            print("Error: fruit selection error")
            print(sel08.center)
            print()
            return

        self.connections.add(sel08.id)
        self.updateCloudCrop()
        self.ask_selection = False

    def sendSocketPose(self):
        # echo-client.py
        pose = Pose(self.view_pose)
        pose_str = str(pose)
        self.socket.sendall(pose_str.encode())
