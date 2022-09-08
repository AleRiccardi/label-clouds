import numpy as np
import matplotlib.pyplot as plt

from typing import List
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from label.types.fruit import Fruit
from label.types.pose import Pose
from label.utils.commons import pick_points, crop_cloud
from label.utils.tools import set_axes_equal, print_title


class AreaGT:
    NUM_LANES = 4

    def __init__(self, proportion=0.25):
        #  print_title("Area GT")
        self.x_min = 0
        self.x_max = 0
        self.y_middle = 0
        self.proportion = proportion

    def fromPoints(self, point1, point2):
        points = []
        points.append(point1)
        points.append(point2)
        points = np.array(points)

        self.x_min = points[:, 0].min()
        self.x_max = points[:, 0].max()
        self.y_middle = points[:, 1].mean()

        #  self.print()

    def fromPointsPick(self, cloud):
        idxs_points = pick_points(cloud)
        points = []
        points.append(cloud.points[idxs_points[0]])
        points.append(cloud.points[idxs_points[1]])
        points = np.array(points)
        self.computeGT(points)

    def computeGT(self, points):
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        self.y_middle = points[:, 1].mean()

        # X - axis
        proportion_in_meters = (x_max - x_min) * self.proportion
        half = (x_max - x_min) / 2

        start = half + x_min
        end = half + proportion_in_meters + x_min

        self.x_min = start
        self.x_max = end
        #  self.print()

    def print(self):
        print(" - area estimated:")
        print(
            " * start(X-axis): {}\n * stop(X-axis): {}\n * middle point(Y-axis): {}".format(
                self.x_min, self.x_max, self.y_middle
            )
        )

    def isGT(self, pose: Pose):
        if self.x_min < pose.getPosition()[0] < self.x_max:
            if pose.getPosition()[1] < self.y_middle:
                if 70 < pose.getEuler(True)[2] < 110:
                    return True
            if pose.getPosition()[1] > self.y_middle:
                if -110 < pose.getEuler(True)[2] < -70:
                    return True
        return False

    def cropCloud(self, cloud):
        #  print(" - cropping cloud area")
        x_middle = (self.x_max + self.x_min) / 2
        position_middle = [x_middle, self.y_middle, 0]
        size_area = self.x_max - self.x_min

        T = np.eye(4)
        T[:-1, -1] = position_middle

        cloud_copy = deepcopy(cloud)
        cloud_copy.transform(np.linalg.inv(T))
        cloud_crop = crop_cloud(cloud_copy, size_area)
        cloud_crop.transform(T)

        return cloud_crop

    def cropFruitsByLane(self, fruits: List[Fruit], mean_lane_gt: np.ndarray):
        # Get GT area fruits
        fruits_area = self.cropFruits(fruits)
        # Select GT lane
        fruits_out = self.selectGTLane(fruits_area, mean_lane_gt)

        return fruits_out

    def cropFruits(self, fruits: List[Fruit]):
        # Get GT area fruits
        fruits_out: List[Fruit] = []
        for fruit in fruits:
            if self.x_min < fruit.getPose().getPosition()[0] < self.x_max:
                fruits_out.append(fruit)

        return fruits_out

    def selectGTLane(self, fruits: List[Fruit], mean_lane_gt: np.ndarray):
        # Prepare data
        positions = np.array([f.getPose().getPosition() for f in fruits])
        positions_pca = self.pcaLanes(positions)

        # Cluster the lanes
        ag = AgglomerativeClustering(n_clusters=self.NUM_LANES)
        labels = ag.fit_predict(positions_pca)

        # Split the fruits in lanes
        lanes_to_idxs = {}
        lanes_to_positions = {}
        for idx, label in enumerate(labels):
            position = positions[idx]
            if label not in lanes_to_idxs:
                lanes_to_idxs[label] = [idx]
                lanes_to_positions[label] = [position]
            lanes_to_idxs[label].append(idx)
            lanes_to_positions[label].append(position)

        # Get the correct lane, based on the mean position of the GT lane
        means = np.array(
            [np.array(poss).mean(axis=0) for poss in lanes_to_positions.values()]
        )
        idx_lane_gt = np.argmin(np.abs((means - mean_lane_gt))[:, 1])
        idxs_fruits_lane_gt = list(lanes_to_idxs.values())[idx_lane_gt]

        # Get the fruits of the selected lane
        fruits_out: List[Fruit] = []
        for idx, fruit in enumerate(fruits):
            if idx in idxs_fruits_lane_gt:
                fruits_out.append(fruit)

        return fruits_out

    def pcaLanes(self, data: np.ndarray):
        mean = data.mean(axis=0)
        mean = np.expand_dims(mean, axis=0)
        data_mean = data - mean

        # PCA with SkLearn (working)
        pca = PCA(3)
        proj_pca = pca.fit_transform(data_mean)
        # Keep the second component which is perpendicolar to the direction of
        # the lanes, this will allow to analyze and suddivide the lanes in 4
        # lanes.
        PC2 = proj_pca[:, 1]
        PC1 = np.zeros(PC2.shape)
        PC3 = np.zeros(PC1.shape)
        data_pca = np.array([PC1, PC2, PC3]).T

        #  fig = plt.figure()
        #  fig.suptitle('INV', fontsize=16)
        #  ax_inv = fig.add_subplot(projection='3d')
        #  ax_inv.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c="b")
        #  plt.xlabel('PC 1')
        #  plt.ylabel('PC 2')
        #  set_axes_equal(ax_inv)
        #  plt.show()

        return data_pca
