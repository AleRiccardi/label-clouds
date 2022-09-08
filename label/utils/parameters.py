import os
import yaml
from munch import DefaultMunch, Munch


class Parameters:
    def __init__(self, path) -> None:
        with open(path, "rb") as f:
            self.params = yaml.safe_load(f.read())

        self.params_dict = {}
        self.params_dict.update(self.params)

    def get(self) -> Munch:
        return DefaultMunch.fromDict(self.params_dict)
