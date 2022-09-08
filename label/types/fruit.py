import random
import open3d as o3d
import numpy as np

from typing import Dict

from label.types.camera import Camera
from label.types.pose import Pose
from label.types.bounding_box import BoundingBox
from label.utils.tools import (
    image_to_world,
    read_image,
    world_to_image,
)


class Fruit:
    DEPTH_STD = 0.35
    DEPTH_MIN = 0.1
    DEPTH_MAX = 1.5

    def __init__(
        self,
        pose: Pose = Pose(),
        shape: np.ndarray = np.array([0.0, 0.0]),
        color: np.ndarray = np.array([1.0, 0.0, 0.0]),
        track_id: int = 0,
    ) -> None:
        self.track_id: int = track_id
        self.setPose(pose)
        self.setShape(shape)
        self.setColor(color)

    def isInitialized(self) -> bool:
        return not self.pose.isIdentity()

    def setPose(self, pose: Pose) -> None:
        self.pose = pose

    def setShape(self, shape: np.ndarray) -> None:
        self.shape = np.array(shape)
        self.height = shape[0]
        self.width = shape[1]

    def setColor(self, color: np.ndarray) -> None:
        self.color = np.array(color)

    def getPose(self) -> Pose:
        return self.pose

    def getShape(self) -> np.ndarray:
        return self.shape

    def getColor(self) -> np.ndarray:
        return self.color

    def getRadius(self) -> float:
        return float(self.shape.min() * 0.8)

    def getMesh(self, T=np.eye(4), color_in=None) -> o3d.geometry.TriangleMesh:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.width / 2)
        mesh_sphere.compute_vertex_normals()

        color = sum(self.color) / 3
        color = [color, color, color]
        if color_in:
            color = color_in
        mesh_sphere.paint_uniform_color(color_in)
        pose = self.getPose().compose(T)
        mesh_sphere.translate(pose.getPosition())

        return mesh_sphere

    def fromDict(self, data: Dict[str, object]):
        self.track_id = int(data["track_id"])
        self.pose = Pose(position=np.array(data["position"]))
        self.shape = np.array(data["shape"])
        self.height = self.shape[0]
        self.width = self.shape[1]
        self.color = np.array(data["color"])
        return self

    def toDict(self) -> Dict[str, object]:
        data = {
            "track_id": self.track_id,
            "position": self.pose.getPosition().tolist(),
            "shape": self.shape.tolist(),
            "color": self.color.tolist(),
        }
        return data

    def clone(self, fruit):
        if not isinstance(fruit, Fruit):
            raise TypeError

        self.track_id = fruit.track_id
        self.pose = fruit.pose
        self.height = fruit.height
        self.width = fruit.width
        self.color = fruit.color

    def __eq__(self, fruit):
        if not isinstance(fruit, Fruit):
            raise TypeError
        if not hasattr(fruit, "track_id"):
            raise NotImplemented
        return self.track_id == fruit.track_id

    def __hash__(self):
        return self.track_id


class FruitInImage(Fruit):
    def __init__(self, bnb: BoundingBox, image) -> None:
        Fruit.__init__(self)

        self.image = image
        self.camera = Camera(self.image.getOrientation())
        self.bnb: BoundingBox = bnb
        self.bnb_projection: BoundingBox
        self.depth_cur = 0

    def getProjection(self) -> BoundingBox:
        if self.bnb_projection:
            return self.bnb_projection

        return self.bnb

    def estimate(self, fruit_prev) -> None:
        self.estimatePose(fruit_prev)
        self.estimateShape()
        #  self.estimateColor()

    def estimatePose(self, fruit_prev=None):
        self.depth_cur = self.getDepthFromDisparity(fruit_prev)
        self.pose = self.getGlobalPose(self.depth_cur)

    def estimateShape(self) -> None:
        X_tl = np.array([self.bnb.x_min, self.bnb.y_min, 1])
        X_bl = np.array([self.bnb.x_max, self.bnb.y_max, 1])
        T_cam = self.image.pose_camera.getMatrix()
        depth = self.getDepth()

        X_left_w = image_to_world(X_tl, self.camera.K, T_cam, depth)
        X_right_w = image_to_world(X_bl, self.camera.K, T_cam, depth)
        residual = np.absolute(X_left_w - X_right_w)

        # Remember right hand coordinate frames
        height = residual[2]
        width = residual[0]

        self.shape = np.array([height, width])

    def estimateColor(self):
        y_shift = random.randint(-5, 5)
        x_shift = random.randint(-5, 5)
        image = read_image(self.image.path_image)
        color = image[self.bnb.y_cent + y_shift, self.bnb.x_cent + x_shift] / 255
        self.color = color

    def getGlobalPose(self, depth):
        X_image = self.bnb.getCenter()
        T_cam_cur = self.image.pose_camera.getMatrix()

        X_world = image_to_world(X_image, self.camera.K, T_cam_cur, depth)
        pose = Pose(position=X_world[:3])

        return pose

    def getDepthFromDisparity(self, fruit_prev):
        if not fruit_prev:
            return self.DEPTH_STD

        # Baseline
        t_cam_cur = self.image.pose_camera.getPosition()
        t_cam_prev = fruit_prev.image.pose_camera.getPosition()
        baseline = np.linalg.norm(t_cam_cur - t_cam_prev)

        # Disparity
        x_cur = self.bnb.getCenter()[0]
        x_prev = fruit_prev.bnb.getCenter()[0]
        disparity = x_cur - x_prev

        depth = baseline * self.camera.focal / (disparity + 0.0001)
        depth = (
            self.DEPTH_STD
            if depth < self.DEPTH_MIN or depth > self.DEPTH_MAX
            else depth
        )

        return depth

    def getDepth(self):
        X = self.getPose().getPosition()
        T_cam = self.image.pose_camera.getMatrix()
        x_image = world_to_image(X, self.camera.K, T_cam, homogeneous=True)

        return x_image[2]
