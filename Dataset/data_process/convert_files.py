import pandas as pd
import os

src_dir = "/CAShift/Dataset/data_process/test/output/train"

for filename in os.listdir(src_dir):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(src_dir, filename))
        index = filename.split("_")[1].split(".")[0]
        df['Label'] = f"normal_{index}"
        df.to_csv(os.path.join(src_dir, filename), index=False)
