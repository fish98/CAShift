import pandas as pd
import os

all_file_path = "/CAShift/Dataset/data_process/test/output/train"
dataframes = []

for filename in os.listdir(all_file_path):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(all_file_path, filename))
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv("combined.csv", index=False)