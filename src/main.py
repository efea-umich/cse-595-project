from os import PathLike

import pandas as pd
import yaml

from basketball_config import BasketballConfig

import fire

from enhance_data import DataEnhancer


class Main:
    def train_model(self, config_path: PathLike, data_path: PathLike):
        with open(config_path, "r") as f:
            config = BasketballConfig.model_validate(yaml.safe_load(f))

        data = pd.read_csv(data_path)
        data = DataEnhancer(data, config).enhance_data(data)

        print(data.head())


if __name__ == "__main__":
    fire.Fire(Main)
