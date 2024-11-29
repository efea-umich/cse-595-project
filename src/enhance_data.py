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
        data["PlayType"] = self.get_play_type(data)
        data["PlaySide"] = self.get_play_side(data)
        data["TimeoutSide"] = self.get_timeout_team(data)
        data["WinningTeam"] = self.get_winning_team(data)

        data = self.remove_unused_columns(data)

        return data

    def remove_unused_columns(self, data: pd.DataFrame):
        columns_to_remove = self.config.kept_cols
        data = data[columns_to_remove]
        return data

    def get_timeout_team(self, df: pd.DataFrame):
        timeout_side = pd.Series(
            np.where(
                df["TimeoutTeam"] == df["HomeTeam"],
                0,
                np.where(df["TimeoutTeam"] == df["AwayTeam"], 1, np.nan),
            )
        )

        invalid_rows = df["TimeoutTeam"].notnull() & pd.isna(timeout_side)
        if invalid_rows.any():
            row = df[invalid_rows]
            logger.error("Invalid rows found when determining TimeoutSide: {}", row)
            raise ValueError(
                "Invalid rows found: TimeoutTeam is not null but TimeoutSide could not be determined. "
                "Ensure TimeoutTeam matches HomeTeam or AwayTeam."
            )

        return timeout_side

    def get_winning_team(self, df: pd.DataFrame):
        winning_team = pd.Series(
            np.where(
                df["WinningTeam"] == df["HomeTeam"],
                0,
                np.where(df["WinningTeam"] == df["AwayTeam"], 1, np.nan),
            )
        )

        if pd.isnull(winning_team).any():
            row = df[pd.isnull(winning_team)]
            logger.error("WinningTeam does not match HomeTeam or AwayTeam: {}", row)
            raise ValueError("WinningTeam does not match HomeTeam or AwayTeam")
        return winning_team

    def get_play_side(self, df: pd.DataFrame) -> pd.Series:
        play_side = pd.Series(
            np.where(
                pd.notnull(df["HomePlay"]),
                0,
                np.where(pd.notnull(df["AwayPlay"]), 1, np.nan),
            )
        )

        return play_side



    def get_play_type(self, data: pd.DataFrame) -> pd.Series:
        play_type_col = pd.Series(np.nan, index=data.index, dtype=object)
        for play_type, column in self.config.play_types.items():
            mask = data[column].notnull() & play_type_col.isna()
            play_type_col[mask] = play_type
        return play_type_col