import json
import random
import numpy as np

from typing import Dict, List

from label.types.pose import Pose


class Selection:
    RADIUS_FACTOR = 1.1

    def __init__(
        self,
        id_=-1,
        center=np.array([0, 0, 0]),
        radius: float = 0,
        color_mean: List[float] = [1, 0, 0],
        color_std: List[float] = [0, 0, 0],
    ) -> None:
        self.id = id_
        self.center: np.ndarray = center
        self.radius = radius * self.RADIUS_FACTOR
        self.color_mean = color_mean
        self.color_std = color_std
        self.color_display = [random.uniform(0, 1) for _ in range(3)]

    def isEmtpy(self):
        return (self.center == np.array([0, 0, 0])).all()

    def fromDict(self, data, T=np.eye(4)):
        self.id = data["id"]
        center = np.array(data["center"])
        center_pose = Pose(position=center)
        center_pose = center_pose.compose(T)
        self.center = center_pose.getPosition()
        self.radius = data["radius"]
        self.color_mean = data["color_mean"]
        self.color_std = data["color_std"]
        self.color_display = data["color_display"]
        return self

    def toDict(self, T=np.eye(4)) -> Dict[str, object]:
        center_pose = Pose(position=self.center)
        center_pose = center_pose.compose(np.linalg.inv(T))
        center_list = center_pose.getPosition().tolist()
        data = {
            "id": self.id,
            "center": center_list,
            "radius": self.radius,
            "color_mean": self.color_mean,
            "color_std": self.color_std,
            "color_display": self.color_display,
        }
        return data


class Selections:
    def __init__(self, T=np.eye(4)) -> None:
        self.T = T
        self.id_incr: int = 0
        self.selections: Dict[int, Selection] = {}

    def add(self, center, radius, color_mean, color_std):
        sel = Selection(self.id_incr, center, radius, color_mean)
        self.selections[self.id_incr] = sel
        self.id_incr += 1

    def get(self) -> List[Selection]:
        return list(self.selections.values())

    def getClosest(self, position, dist_max=0.1) -> Selection:
        dist = np.inf
        selection = Selection(self.T)

        for sel in self.selections.values():
            dist_cur = np.linalg.norm(position - sel.center)
            if dist_cur < dist and dist_cur < dist_max:
                dist = dist_cur
                selection = sel

        return selection

    def removeLast(self):
        self.id_incr -= 1
        del self.selections[self.id_incr]

    def removeClose(self, position):
        selection = self.getClosest(position)

        if not selection:
            return False

        del self.selections[selection.id]
        return True

    def fromJson(self, json_str):
        data = dict(json.loads(json_str))
        for _, data in data.items():
            sel = Selection().fromDict(data, self.T)
            self.selections[sel.id] = sel
            self.id_incr = sel.id
        self.id_incr += 1
        return self

    def toJson(self) -> str:
        data = {}
        for id_, sel in self.selections.items():
            data[id_] = sel.toDict(self.T)
        json_str = json.dumps(data)
        return json_str

    def __len__(self):
        return len(self.selections)
