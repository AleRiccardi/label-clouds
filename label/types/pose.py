from __future__ import annotations

import math
import numpy as np

from copy import copy
from typing import List
from scipy.spatial.transform import Rotation as R


class Pose:
    def __init__(
        self,
        matrix: np.ndarray = None,
        rotation: np.ndarray = None,
        position: np.ndarray = None,
        timestamp: int = 0,
    ) -> None:
        self.timestamp = int(timestamp)
        self._pose = np.eye(4)

        if matrix is not None:
            self._pose = matrix
            return

        if rotation is not None:
            self._pose[:3, :3] = rotation

        if position is not None:
            self._pose[:3, 3] = position

    def setMatrix(self, matrix: np.ndarray) -> None:
        self._pose = matrix

    def setRotation(self, rotation: np.ndarray) -> None:
        self._pose[:3, :3] = rotation

    def setPosition(self, position: np.ndarray) -> None:
        self._pose[:3, 3] = position

    def getMatrix(self) -> np.ndarray:
        return self._pose

    def getList(self) -> List[float]:
        return self._pose.reshape(-1).tolist()

    def getRotation(self) -> np.ndarray:
        return self._pose[:3, :3]

    def getEuler(self, degree=False):
        R = self.getRotation()
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # [Roll, pith, yaw]
        euler = np.array([x, y, z])
        if degree:
            euler *= 180 / np.pi
        return euler

    def getQuaternion(self):
        r = R.from_matrix(self.getRotation())
        return r.as_quat()

    def getPosition(self) -> np.ndarray:
        return self._pose[:3, 3]

    def isIdentity(self) -> bool:
        return (self._pose == np.eye(4)).all()

    def compose(self, T: np.ndarray) -> Pose:
        new = copy(self)
        new._pose = T @ self._pose
        return new

    def fromStr(self, string: str):
        mtx = np.fromstring(string, sep=" ")
        if mtx.shape[0] == 16:
            mtx = mtx.reshape(4, 4)
            self.setMatrix(mtx)

        return self

    def __str__(self) -> str:
        str_ = str(self.getList())
        # Remove commas and parenthesis
        str_ = str_.replace(",", "")
        str_ = str_.replace("[", "")
        str_ = str_.replace("]", "")

        return str_
