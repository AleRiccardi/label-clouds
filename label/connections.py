import json
import random

from typing import Dict, List


class Connection:
    COLOR_REMOVED = [1, 0, 0]

    def __init__(self, first=-1, second=-1) -> None:
        self.first = first
        self.second = second

        if self.second != -1:
            self.color = [random.uniform(0, 1) for _ in range(3)]
        else:
            self.color = self.COLOR_REMOVED

    def isConnected(self):
        if self.first != -1 and self.second != -1:
            return True
        return False

    def isRemoved(self):
        if self.first != -1 and self.second == -1:
            return True
        return False

    def isEmpty(self):
        if self.first == -1 and self.second == -1:
            return True
        return False

    def fromDict(self, data):
        self.first = data["first"]
        self.second = data["second"]
        self.color = data["color"]
        return self

    def toDict(self) -> Dict[str, int]:
        data = {
            "first": self.first,
            "second": self.second,
            "color": self.color,
        }
        return data

    def __eq__(self, __o: object) -> bool:
        return self.first == __o.first and self.second == __o.second


class Connections:
    MAP_FIRST = "first"
    MAP_SECOND = "second"

    def __init__(self) -> None:
        self.connections: List[Connection] = []
        self.connections_first_conn: Dict[int, Connection] = {}
        self.connections_second_conn: Dict[int, Connection] = {}
        self.connections_first_second: Dict[int, int] = {}
        self.connections_second_first: Dict[int, int] = {}

    def add(self, id_first, id_second=-1):
        conn = Connection(id_first, id_second)
        self.addConnection(conn)

    def addConnection(self, conn: Connection):
        self.connections.append(conn)
        self.connections_first_conn[conn.first] = conn

        if conn.second != -1:
            self.connections_second_conn[conn.second] = conn
            self.connections_first_second[conn.first] = conn.second
            self.connections_second_first[conn.second] = conn.first

    def getList(self) -> List[Connection]:
        return self.connections

    def getConnectionFromFirstId(self, first: int) -> Connection:
        if first in self.connections_first_conn:
            return self.connections_first_conn[first]
        return None

    def getConnectionFromSecondId(self, second: int) -> Connection:
        if second in self.connections_second_conn:
            return self.connections_second_conn[second]
        return None

    def getFirstDict(self) -> Dict[int, int]:
        return self.connections_first_second

    def getSecondDict(self) -> Dict[int, int]:
        return self.connections_second_first

    def getIdsColor(self, which):
        ids_color = {-1: [0.0, 0.0, 0.0]}

        for conn in self.connections:
            if which == self.MAP_FIRST:
                ids_color[conn.first] = conn.color
            elif which == self.MAP_SECOND:
                ids_color[conn.second] = conn.color

        return ids_color

    def removeLast(self):
        self.removeConnection(self.connections[-1])

    def remove(self, id_map, which):
        conn = None

        if which == self.MAP_FIRST:
            if id_map in self.connections_first_conn:
                conn = self.connections_first_conn[id_map]

        elif which == self.MAP_SECOND:
            if id_map in self.connections_second_conn:
                conn = self.connections_second_conn[id_map]

        if conn:
            self.removeConnection(conn)

    def removeConnection(self, conn: Connection):
        idx_del = -1
        for idx, conn_loop in enumerate(self.connections):
            if conn_loop == conn:
                idx_del = idx
                break

        del self.connections[idx_del]
        del self.connections_first_conn[conn.first]
        if conn.second != -1:
            del self.connections_second_conn[conn.second]
            del self.connections_first_second[conn.first]
            del self.connections_second_first[conn.second]

    def fromJson(self, json_type):
        data = list(json_type)
        for conn_dict in data:
            conn_dict = dict(conn_dict)
            conn = Connection().fromDict(conn_dict)
            self.addConnection(conn)
        return self

    def toJson(self) -> str:
        data = []
        for conn in self.connections:
            data.append(conn.toDict())
        json_str = json.dumps(data)
        return json_str

    def __len__(self):
        return len(self.connections)
