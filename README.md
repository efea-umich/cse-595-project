# CSE 595 Project

NBA play-by-play data link [here](https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019)

## Running Instructions

Install UV from [here](https://github.com/astral-sh/uv)

Run `uv sync`

Run 
```bash
$ uv run mlflow server
```

On a separate terminal, run
```bash
$ uv run src/main.py train_model --config_path=play-types.yaml --data_path=data/NBA_PBP_2015-16.csv
```