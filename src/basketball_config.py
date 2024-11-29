from typing import Literal

import pydantic
from pydantic import BaseModel


class BasketballConfig(BaseModel):
    play_types: dict[str, str]
    column_types: dict[str, Literal["numeric", "categorical"]]
    kept_cols: list[str]
