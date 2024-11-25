import numpy as np
import pandas as pd
import yaml

from loguru import logger

class DataEnhancer:
    def __init__(self, data: pd.DataFrame, config_path: str):
        self.data = data
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def enhance_data(self, data: pd.DataFrame):
        data['PlayType'] = data.apply(self.get_play_type, axis=1)
        data['PlaySide'] = data.apply(self.get_play_side, axis=1)
        data['TimeoutSide'] = data.apply(self.generalize_timeout_team, axis=1)

        data = self.remove_unused_columns(data)

        return data

    def remove_unused_columns(self, data: pd.DataFrame):
        columns_to_remove = self.config['KeptCols']
        data = data[columns_to_remove]
        return data

    def generalize_timeout_team(self, row: pd.Series):
        if pd.isnull(row['TimeoutTeam']):
            return np.nan

        if row['TimeoutTeam'] == row['HomeTeam']:
            return 'Home'
        elif row['TimeoutTeam'] == row['AwayTeam']:
            return 'Away'
        else:
            raise ValueError(f'TimeoutTeam does not match HomeTeam or AwayTeam')

    def get_play_side(self, row: pd.Series):
        if pd.notnull(row['HomePlay']):
            return 'Home'
        elif pd.notnull(row['AwayPlay']):
            return 'Away'
        else:
            return np.nan

    def get_play_type(self, row: pd.Series):
        play_types = self.config['PlayTypes']
        for key, value in play_types.items():
            if pd.notnull(row[value]):
                return key

        return np.nan
