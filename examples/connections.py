#! /usr/bin/env python3


from open3d import os
from label.connections_manager import ConnectionsManager
from label.utils.parameters import Parameters


def main():

    params = Parameters(os.environ["LABEL_PATH"] + "configs/parameters/data.yaml").get()
    connections = ConnectionsManager(params)

    while True:
        user_continue = connections.userSelection()

        if not user_continue:
            # Exit
            return

        connections.selectConnection()
        connections.saveConnections()


if __name__ == "__main__":
    main()
