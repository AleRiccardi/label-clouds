#! /usr/bin/env python3

import click

from label.selections_manager import SelectionsManager
from label.utils.parameters import Parameters


@click.command()
@click.argument("path_config")
def main(path_config):
    params = Parameters(path_config).get()
    selection = SelectionsManager(
        params.path_cloud_1,
        params.transformations.gt_1,
        params.path_selections_1,
        "1",
        params,
        False,
    )

    while True:
        user_continue = selection.userSelection()

        if not user_continue:
            # Exit
            return

        selection.selectFruit()
        selection.saveSelections()


if __name__ == "__main__":
    main()
