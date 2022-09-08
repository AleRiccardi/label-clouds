import cv2
import numpy as np

from typing import  List
from label.types.bounding_box import BoundingBox
from label.types.detections import Detections
from label.types.pose import Pose
from label.utils.tools import display_image, read_image


class Image:
    WIDTH = 640
    HEIGHT = 480

    def __init__(
        self, timestamp: int = 0, pose_camera: Pose = Pose(), path_image: str = ""
    ):
        self.empty = False
        if timestamp == 0:
            self.empty = True

        self.timestamp = timestamp
        self.pose_camera = pose_camera
        self.path_image = path_image

    def isEmpty(self):
        return self.empty

    def getOrientation(self) -> str:
        left = "left"
        right = "right"

        if left in self.path_image:
            return left
        elif right in self.path_image:
            return right
        else:
            return "unknown"

    def isUpsideDown(self):
        roll = self.pose_camera.getEuler()[0]
        return abs(abs(roll) - abs(np.pi)) < 0.3

    def _img(self):
        img = read_image(self.path_image)
        return img

    def _img180(self):
        img = self._img()
        img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    def getImg(self, correct=False):
        if correct and self.isUpsideDown():
            return self._img180()
        return self._img()

    def displayImage(self, correct=True):
        image = self.getImg(correct)
        display_image(image)


class ImageDetections(Image):
    def __init__(
        self, timestamp: int, pose_camera: Pose, path_image: str, detections: Detections
    ) -> None:
        Image.__init__(self, timestamp, pose_camera, path_image)
        self.detections = detections
        self.path_detection = detections.path
        self.boundings: List[BoundingBox] = detections.boundings

    def displayDetections(self):
        image = self.getImg()
        for box in self.boundings:
            cv2.rectangle(
                image, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 255, 0), 2
            )
        display_image(image)

    def getCorrectBounding(self, bnb: BoundingBox):
        if self.isUpsideDown():
            return bnb.getRotate180()
        return bnb.get()

    def getImgDetections(self, tracks, display_depth=False):
        img = self.getImg(True)

        for track in tracks:
            fruit = track.getLastFruit()

            if not track.getProjection().isEmpty():
                (y_min, x_min, y_max, x_max) = self.getCorrectBounding(
                    track.getProjection()
                )
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), track.color_dark, 1)

            if fruit.image.timestamp == self.timestamp:
                bb_cur = fruit.bnb
                (y_min, x_min, y_max, x_max) = self.getCorrectBounding(bb_cur)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), track.color, 2)
                # Depth
                if display_depth:
                    depth = round(track.getDepth(self.pose_camera.getMatrix()), 2)
                    cv2.putText(
                        img,
                        str(depth),
                        (x_min + 3, y_max - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        200,
                        1,
                    )
                # ID
                cv2.putText(
                    img,
                    str(track.id),
                    (x_min + 2, y_min + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    200,
                    1,
                )

        return img


