import json
import os
import pathlib

import pandas as pd


def class_to_name():
    current_path = pathlib.Path(__file__).parent.resolve().parent
    data_path = os.path.join(current_path, "data", "ESC-50", "meta", "esc50.csv")
    audio_df = pd.read_csv(data_path)
    map_dict = {}
    for idx, row in audio_df.iterrows():
        map_dict[str(row["target"])] = [str(row["target"]), str(row["category"])]

    return map_dict


if __name__ == "__main__":
    sample = class_to_name()
    with open("index_to_name.json", "w") as fp:
        json.dump(obj=sample, fp=fp, sort_keys=True, indent=4, ensure_ascii=True)
