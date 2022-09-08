import socket
import matplotlib.pyplot as plt
import numpy as np

from typing import List
from label.loader import Loader
from label.types.image import Image
from label.types.pose import Pose


class ImageView:
    HOST = "127.0.0.1"
    PORT = 9999

    def __init__(self, loader: Loader) -> None:
        self.images1 = loader.images1
        self.images2 = loader.images2

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.HOST, self.PORT))
        self.socket.listen()
        self.conn, self.addr = self.socket.accept()
        print("Waiting view pose form {}".format(self.addr))

    def checkConnection(self):
        try:
            self.conn.sendall("test".encode())
            return True
        except:
            return False

    def listen(self):
        data = self.conn.recv(1024).decode()
        pose = Pose().fromStr(str(data))
        if not pose.isIdentity():
            image1, image2 = self.getClosestImages(pose)
            display_two_images(image1, image2)

    def getClosestImages(self, pose):
        image1 = self.getClosestImage(pose, self.images1)
        image2 = self.getClosestImage(pose, self.images2)

        return image1, image2

    def getClosestImage(self, pose: Pose, images: List[Image]) -> Image:
        image_close = Image()
        dist_close = np.Inf

        for image in images:
            dist = self.computeDist(pose, image)

            if dist < dist_close:
                image_close = image
                dist_close = dist

        print(pose.getMatrix())
        print(image_close.pose_camera.getMatrix())

        return image_close

    def computeDist(self, pose: Pose, image):
        # Position difference
        norm = np.linalg.norm(pose.getPosition() - image.pose_camera.getPosition())
        # Angle difference
        yaw1 = pose.getEuler()[2]
        yaw2 = image.pose_camera.getEuler()[2]
        # To degrees
        yaw1 *= 180 / np.pi
        yaw2 *= 180 / np.pi
        angle = abs(yaw1 - yaw2)

        dist = norm * 0.92 + angle * 0.08
        return dist


def display_two_images(image1: Image, image2: Image, block=True):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    fig = plt.figure(figsize=(20, 10))

    fig.add_subplot(1, 2, 1)
    plt.imshow(image1.getImg(True))
    plt.title("Session 1 - ({})".format(image1.timestamp))
    plt.axis("off")

    fig.add_subplot(1, 2, 2)
    plt.imshow(image2.getImg(True))
    plt.title("Session 2 - ({})".format(image2.timestamp))

    plt.show(block=block)
    #  plt.pause(0.000000000001)
