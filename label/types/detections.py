import os
import cv2
import numpy as np
from label.types.bounding_box import BoundingBox
from label.utils.tools import display_image


class Detections:
    def __init__(self, name, shape) -> None:
        self.name = name
        self.shape = shape
        self.boundings = []
        self.path = ""

    def loadtxt(self, path):
        self.path = path
        boundings_np = load_numpy(path)

        if len(boundings_np) == 0:
            return
        if boundings_np.ndim == 1:
            bb = BoundingBox(self.shape)
            bb.loadYolo(boundings_np)
            self.boundings.append(bb)
            return

        for y in range(boundings_np.shape[0]):
            bb = BoundingBox(self.shape)
            bb.loadYolo(boundings_np[y, :])
            self.boundings.append(bb)

        self.checkInsidePerimeter()

    def checkInsidePerimeter(self):
        boundings_correct = []
        for bnb in self.boundings:

            height_margin = self.shape[0] * 0.04
            width_margin = self.shape[1] * 0.04

            if (
                width_margin < bnb.x_cent < self.shape[1] - width_margin
                and height_margin < bnb.y_cent < self.shape[0] - height_margin
            ):
                boundings_correct.append(bnb)

        self.boundings = boundings_correct

    def save(self, path_save):
        if len(self.boundings) == 0:
            return

        if not os.path.exists(path_save):
            os.makedirs(path_save)
        path_file_out = os.path.join(path_save, self.name + ".txt")

        print(self.__str__(), file=open(path_file_out, "w"))

    def display(self, img):
        for box in self.boundings:
            cv2.rectangle(
                img, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 255, 0), 1
            )
        display_image(img, self.name)

    def __str__(self):
        boundings_str = []

        # For each bounding box
        for bnb in self.boundings:

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            center_x = (bnb.x_min + bnb.x_max) / 2
            center_y = (bnb.y_min + bnb.y_max) / 2
            width = bnb.x_max - bnb.x_min
            height = bnb.y_max - bnb.y_min

            # Normalise the co-ordinates by the dimensions of the image
            image_h, image_w = self.shape
            center_x /= image_w
            center_y /= image_h
            width /= image_w
            height /= image_h

            # Write the bbox details to the file
            boundings_str.append(
                "{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                    0, bnb.accuracy, center_x, center_y, width, height
                )
            )

        string = "\n".join(boundings_str)
        return string


def load_numpy(path):
    f = open(path, "r")
    data = []
    for line in f.readlines():
        line = line.strip()
        data.append(line.split(" "))
    f.close()
    return np.array(data, dtype=float)
