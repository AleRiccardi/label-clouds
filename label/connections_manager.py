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
from label.utils.commons import add_clouds
from label.utils.tools import pick_w_pinhole, print_title, pick_points
from label.utils.user import user_input


class ConnectionsManager:
    TRANSFORM_2 = np.eye(4)
    TRANSFORM_2[:-1, -1] = [0, 1.2, 0]
    TRANSFORM_CLOSE_2 = np.eye(4)
    TRANSFORM_CLOSE_2[:-1, -1] = [0, 0.4, 0]
    HOST = "127.0.0.1"
    PORT = 9999

    def __init__(self, params) -> None:
        self.params = params
        self.view_pose = np.eye(4)
        self.clear = lambda: os.system("clear")
        self.pinhole = None
        self.initMessage()

        self.path_connections = params.path_connections
        path, _ = os.path.split(self.path_connections)

        if not os.path.exists(path):
            os.makedirs(path)

        self.connections: Connections = Connections()
        self.loadConnections()

        self.selection1 = SelectionsManager(
            params.path_cloud_1,
            params.transformations.gt_1,
            params.path_selections_1,
            "1",
            params,
            False,
        )
        self.selection2 = SelectionsManager(
            params.path_cloud_2,
            params.transformations.gt_2,
            params.path_selections_2,
            "2",
            params,
            False,
        )

        self.cloud_ds_1 = o3d.geometry.PointCloud()
        self.cloud_ds_2 = o3d.geometry.PointCloud()
        self.cloud_crop_1 = o3d.geometry.PointCloud()
        self.cloud_crop_2 = o3d.geometry.PointCloud()
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
        if not self.ask_selection:
            return True

        self.initMessage()
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

        if not self.isViewPointSet() and action != 5:
            self.selectViewPoint()

        if action == 1:
            self.selectHarvestedFruit()
        elif action == 2:
            self.removeConnection()
        elif action == 3:
            self.selectViewPoint()
        elif action == 4:
            self.fruitSelection()
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
        idxs, self.pinhole = pick_w_pinhole(self.cloud_crop, self.pinhole)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 1:
            print("Attention: select only one point")
            return

        sel1: Selection = Selection()
        sel2: Selection = Selection()
        for idx in idxs:

            if idx < len(self.cloud_crop_1.points):
                position = self.cloud_crop_1.points[idx]
                sel1 = self.selection1.getFromPosition(position)
            else:
                idx -= len(self.cloud_crop_1.points)
                position = self.cloud_crop_2.points[idx]
                pose = np.eye(4)
                pose[:-1, -1] = position
                pose = np.linalg.inv(self.TRANSFORM_CLOSE_2) @ pose
                position = pose[:-1, -1].flatten()
                sel2 = self.selection2.getFromPosition(position)

        if not sel1.isEmtpy():
            self.connections.remove(sel1.id, Connections.MAP_FIRST)
        elif not sel2.isEmtpy():
            self.connections.remove(sel2.id, Connections.MAP_SECOND)

        self.updateCloudCrop()
        self.ask_selection = True

    def fruitSelection(self):
        print("Chose:", " 1) Map 1", " 2) Map 2", " 3) Go back", sep=os.linesep)
        action = user_input("")

        if action == 1:
            self.selection1.userSuperSelection()
            self.selection2.updateViewPoint(self.selection1.view_pose)
            self.pinhole = None
            self.ask_selection = True
        elif action == 2:
            self.selection2.userSuperSelection()
            self.selection1.updateViewPoint(self.selection2.view_pose)
            self.pinhole = None
            self.ask_selection = True
        elif action == 3:
            return

    def updateCloudDS(self):
        ids_color_1 = self.connections.getIdsColor(Connections.MAP_FIRST)
        ids_color_2 = self.connections.getIdsColor(Connections.MAP_SECOND)

        self.selection1.updateCloudDS(ids_color_1)
        self.selection2.updateCloudDS(ids_color_2)

        self.cloud_ds_1 = self.selection1.cloud_ds_color
        self.cloud_ds_2 = deepcopy(self.selection2.cloud_ds_color).transform(
            self.TRANSFORM_2
        )

        self.cloud_ds = add_clouds(self.cloud_ds_1, self.cloud_ds_2)

    def updateCloudCrop(self):
        self.selection1.updateViewPoint(self.view_pose)
        self.selection2.updateViewPoint(self.view_pose)

        ids_color_1 = self.connections.getIdsColor(Connections.MAP_FIRST)
        ids_color_2 = self.connections.getIdsColor(Connections.MAP_SECOND)

        self.selection1.updateCloudCrop(ids_color_1)
        self.selection2.updateCloudCrop(ids_color_2)

        self.cloud_crop_1 = self.selection1.cloud_crop
        self.cloud_crop_2 = deepcopy(self.selection2.cloud_crop).transform(
            self.TRANSFORM_CLOSE_2
        )

        self.cloud_crop = add_clouds(self.cloud_crop_1, self.cloud_crop_2)

    def getCloudViewPoint(self, idx):
        if idx < len(self.cloud_ds_1.points):
            position = self.cloud_ds_1.points[idx]
            self.view_pose = np.eye(4)
            self.view_pose[:-1, -1] = position
        else:
            idx -= len(self.cloud_ds_1.points)
            position = self.cloud_ds_2.points[idx]
            self.view_pose = np.eye(4)
            self.view_pose[:-1, -1] = position
            self.view_pose = np.linalg.inv(self.TRANSFORM_2) @ self.view_pose

    def selectViewPoint(self):
        self.updateCloudDS()

        while True:
            idxs = pick_points(self.cloud_ds, "View point selection")
            if len(idxs) == 0:
                self.ask_selection = True
                return

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

    def isViewPointSet(self):
        if (self.view_pose == np.eye(4)).all():
            return False
        return True

    def selectConnection(self):
        # Pick points
        idxs, self.pinhole = pick_w_pinhole(self.cloud_crop, self.pinhole)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 2:
            print("Attention: select only two points")
            return

        sel1: Selection = Selection()
        sel2: Selection = Selection()
        for idx in idxs:

            if idx < len(self.cloud_crop_1.points):
                position = self.cloud_crop_1.points[idx]
                sel1 = self.selection1.getFromPosition(position)
            else:
                idx -= len(self.cloud_crop_1.points)
                position = self.cloud_crop_2.points[idx]
                pose = np.eye(4)
                pose[:-1, -1] = position
                pose = np.linalg.inv(self.TRANSFORM_CLOSE_2) @ pose
                position = pose[:-1, -1].flatten()
                sel2 = self.selection2.getFromPosition(position)

        if sel1.isEmtpy() or sel2.isEmtpy():
            print("Connection error")
            print(sel1.center)
            print(sel2.center)
            return

        self.connections.add(sel1.id, sel2.id)
        self.updateCloudCrop()
        self.ask_selection = False

    def selectHarvestedFruit(self):
        # Pick points
        idxs, self.pinhole = pick_w_pinhole(self.cloud_crop, self.pinhole)
        if len(idxs) == 0:
            self.ask_selection = True
            return
        elif len(idxs) != 1:
            print("Attention: select only one points")
            return

        sel1: Selection = Selection()
        for idx in idxs:
            if idx < len(self.cloud_crop_1.points):
                position = self.cloud_crop_1.points[idx]
                sel1 = self.selection1.getFromPosition(position)
            else:
                print("Error: removed fruit can be only in session 1")

        if sel1.isEmtpy():
            print("Error: fruit selection error")
            print(sel1.center)
            print()
            return

        self.connections.add(sel1.id)
        self.updateCloudCrop()
        self.ask_selection = False

    def sendSocketPose(self):
        # echo-client.py
        pose = Pose(self.view_pose)
        pose_str = str(pose)
        self.socket.sendall(pose_str.encode())
