#! /usr/bin/env python3


import click

from open3d import os
from label.connections_manager import ConnectionsManager
from label.utils.parameters import Parameters


@click.command()
@click.argument("path_config")
def main(path_config):
    params = Parameters(path_config).get()
    connections = ConnectionsManager(params)

    while True:
        user_continue = connections.userSelection()

        if not user_continue:
            # Exit
            return

        if connections.isViewPointSet():
            connections.selectConnection()
            connections.saveConnections()


if __name__ == "__main__":
    main()
