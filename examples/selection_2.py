#! /usr/bin/env python3


from open3d import os
from label.selections_manager import SelectionsManager
from label.utils.parameters import Parameters


def main():

    params = Parameters(os.environ["LABEL_PATH"] + "configs/parameters/data.yaml").get()
    selection = SelectionsManager(
        params.path_cloud_1,
        params.transformations.gt_08,
        params.path_selections_1,
        "08",
        params,
        False,
    )

    selection.userSuperSelection()


if __name__ == "__main__":
    main()
