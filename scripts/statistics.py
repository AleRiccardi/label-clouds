import os
import json
from typing import List
import numpy as np

from label.connections import Connections
from label.selections import Selections


def load_connections(path) -> Connections:
    file = open(path, "r")
    json_type = json.loads(file.read())
    connections = Connections().fromJson(json_type)
    file.close()

    print(" - found {} connections".format(len(connections)))
    return connections


def load_selections(path) -> Selections:
    file = open(path, "r")
    selections = Selections().fromJson(file.read())
    file.close()

    print(" - loaded {} fruits selections".format(len(selections)))
    return selections


def match_connections(connections_08_14: Connections, connections_14_21: Connections):
    matchings = []
    for id_14, id_08 in connections_08_14.getSecondDict().items():
        conn_14_21 = connections_14_21.getConnectionFromFirstId(id_14)
        if conn_14_21:
            matchings.append([id_08, id_14, conn_14_21.second])
    return matchings


def print_statistics(
    matchings: List[List[int]],
    selections_08: Selections,
    selections_14: Selections,
    selections_21: Selections,
):
    for match in matchings:
        id_08, id_14, id_21 = match
        sel_08 = selections_08.get(id_08)
        sel_14 = selections_14.get(id_14)
        sel_21 = selections_21.get(id_21)
        if sel_08 and sel_14 and sel_21:
            print("-" * 150)
            print("MEAN")
            print(sel_08.color_mean)
            print(sel_14.color_mean)
            print(sel_21.color_mean)
            print()
            print("STD")
            print(sel_08.color_std)
            print(sel_14.color_std)
            print(sel_21.color_std)
            print()
            print("VOLUME")
            print(sel_08.volume())
            print(sel_14.volume())
            print(sel_21.volume())
            print()


def main():
    path_base = os.path.join(os.environ["LABEL_PATH"], "configs/data/")
    selections_08 = load_selections(os.path.join(path_base, "08_14/selections_1.json"))
    selections_14 = load_selections(os.path.join(path_base, "08_14/selections_2.json"))
    selections_21 = load_selections(os.path.join(path_base, "14_21/selections_2.json"))
    conns_08_14 = load_connections(os.path.join(path_base, "08_14/connections.json"))
    conns_14_21 = load_connections(os.path.join(path_base, "14_21/connections.json"))

    matchings = match_connections(conns_08_14, conns_14_21)
    print_statistics(matchings, selections_08, selections_14, selections_21)


if __name__ == "__main__":
    main()
