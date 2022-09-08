import numpy as np


class BoundingBox:
    def __init__(self, shape=(0, 0)) -> None:
        self.empty = False
        if shape[0] == 0 and shape[1] == 0:
            self.empty = True

        self.accuracy = 1
        self.img_height = shape[0]
        self.img_width = shape[1]

        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0

        self.x_cent = 0
        self.y_cent = 0
        self.width = 0
        self.height = 0
        self.half_width = 0
        self.half_height = 0

    def isEmpty(self):
        return self.empty

    def get(self):
        return (self.y_min, self.x_min, self.y_max, self.x_max)

    def getRotate180(self):
        y_max = self.img_height - self.y_min
        x_max = self.img_width - self.x_min
        y_min = self.img_height - self.y_max
        x_min = self.img_width - self.x_max
        return (y_min, x_min, y_max, x_max)

    def getArea(self):
        return self.width * self.height

    def load(self, xmin, ymin, xmax, ymax):
        self.x_min = xmin
        self.y_min = ymin
        self.x_max = xmax
        self.y_max = ymax

    def loadYolo(self, data: np.ndarray):
        try:
            if data.shape[0] == 5:
                self.x_cent = int(data[1] * self.img_width)
                self.y_cent = int(data[2] * self.img_height)
                self.width = int(data[3] * self.img_width)
                self.height = int(data[4] * self.img_height)
            elif data.shape[0] == 6:
                self.accuracy = data[1]
                self.x_cent = int(data[2] * self.img_width)
                self.y_cent = int(data[3] * self.img_height)
                self.width = int(data[4] * self.img_width)
                self.height = int(data[5] * self.img_height)

            self.half_height = (self.height // 2) - 1
            self.half_width = (self.width // 2) - 1

            self.y_min = self.y_cent - self.half_height
            self.x_min = self.x_cent - self.half_width
            self.y_max = self.y_cent + self.half_height
            self.x_max = self.x_cent + self.half_width
        except Exception as e:
            print(e)
            print(data)

    def updateCenter(self, X_cent):
        self.x_cent = X_cent[0]
        self.y_cent = X_cent[1]
        self.y_min = self.y_cent - self.half_height
        self.x_min = self.x_cent - self.half_width
        self.y_max = self.y_cent + self.half_height
        self.x_max = self.x_cent + self.half_width

    def getCenter(self, homogenous=True):
        if homogenous:
            return np.array([self.x_cent, self.y_cent, 1], dtype=np.float64)
        else:
            return np.array([self.x_cent, self.y_cent])

    def __str__(self) -> str:
        string = "xmin: {}, xmax: {}, ymin: {}, ymax: {}".format(
            self.x_min, self.x_max, self.y_min, self.y_max
        )
        return string
