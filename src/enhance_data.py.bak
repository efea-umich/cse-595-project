import numpy as np
import pandas as pd
import yaml

from loguru import logger

from basketball_config import BasketballConfig


class DataEnhancer:
    def __init__(self, data: pd.DataFrame, config: BasketballConfig):
        self.data = data
        self.config = config

    def enhance_data(self, data: pd.DataFrame):
        data["PlayType"] = data.apply(self.get_play_type, axis=1)
        data["PlaySide"] = data.apply(self.get_play_side, axis=1)
        data["TimeoutSide"] = data.apply(self.generalize_timeout_team, axis=1)
        data["WinningTeam"] = data.apply(self.generalize_winning_team, axis=1)

        data = self.remove_unused_columns(data)

        return data

    def remove_unused_columns(self, data: pd.DataFrame):
        columns_to_remove = self.config.kept_cols
        data = data[columns_to_remove]
        return data

    def generalize_timeout_team(self, row: pd.Series):
        if pd.isnull(row["TimeoutTeam"]):
            return np.nan

        if row["TimeoutTeam"] == row["HomeTeam"]:
            return "Home"
        elif row["TimeoutTeam"] == row["AwayTeam"]:
            return "Away"
        else:
            raise ValueError(f"TimeoutTeam does not match HomeTeam or AwayTeam")

    def generalize_winning_team(self, row: pd.Series):
        if pd.isnull(row["WinningTeam"]):
            return np.nan

        if row["WinningTeam"] == row["HomeTeam"]:
            return "Home"
        elif row["WinningTeam"] == row["AwayTeam"]:
            return "Away"
        else:
            raise ValueError(f"WinningTeam does not match HomeTeam or AwayTeam")

    def get_play_side(self, row: pd.Series):
        if pd.notnull(row["HomePlay"]):
            return "Home"
        elif pd.notnull(row["AwayPlay"]):
            return "Away"
        else:
            return np.nan

    def get_play_type(self, row: pd.Series):
        play_types = self.config.play_types
        for key, value in play_types.items():
            if pd.notnull(row[value]):
                return key

        return np.nan
