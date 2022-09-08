from label.utils.parameters import Parameters


class Camera(object):
    __instance_left = None
    __instance_right = None

    def __new__(cls, orientation: str = "left", params: Parameters = None):
        if (
            Camera.__instance_left is None
            and Camera.__instance_right is None
            and params is not None
        ):
            Camera.__instance_left = object.__new__(cls)
            Camera.__instance_right = object.__new__(cls)

            Camera.__instance_left.focal = params.calibrations.camera_left.focal
            Camera.__instance_left.K = params.calibrations.camera_left.K
            Camera.__instance_left.T_robot_cam = (
                params.calibrations.camera_left.T_robot_cam
            )

            Camera.__instance_right.focal = params.calibrations.camera_right.focal
            Camera.__instance_right.K = params.calibrations.camera_right.K
            Camera.__instance_right.T_robot_cam = (
                params.calibrations.camera_right.T_robot_cam
            )

        if orientation == "left":
            return Camera.__instance_left
        elif orientation == "right":
            return Camera.__instance_right
