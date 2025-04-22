# Save all log data into seperate chunked files

import os
from glob import glob
from typing import List
from datetime import datetime
from multiprocessing import Pool
import random

import pandas as pd
from sysdig import Sysdig

def get_scap_paths(scap_dir: str) -> List[str]:
    """Get all paths to .scap files
    """
    paths = glob(os.path.join(scap_dir, "**", "*.scap"), recursive=True)
    return paths

def get_scap_dfs(scap_paths: List[str], attack: str, index: int) -> pd.DataFrame:
    """Load .scap files and convert to DataFrame
    """
    pool = Pool()
    scap_dfs = None
    final_dfs = []

    with Pool() as pool:
        async_result = pool.map_async(Sysdig().process_scap, scap_paths, chunksize=50)
        pool.close()
        pool.join()
        scap_dfs = async_result.get()

    for scap_path, scap_df in zip(scap_paths, scap_dfs):

        if all(item in scap_df.columns for item in ['timestamp', 'cpuid', 'syscall']):

            today = datetime.now().strftime('%Y-%m-%d')
            
            # if index == -1 then normal, else attack
            scap_df["label"] = f'-' if attack == 'normal' else f"{attack}_{index}"

            scap_df["timestamp"] = pd.to_datetime(today + " " + scap_df["timestamp"], format="ISO8601")
            scap_df["timestamp"] = (scap_df["timestamp"] - pd.Timestamp(today)).dt.total_seconds()

            final_dfs.append(scap_df)
        else:
            print(f"Warning: incomplete DataFrame {scap_path}")
    final_df = pd.concat(final_dfs)
    final_df.reset_index(drop=True, inplace=True)
    return final_df

columns={
    "label": "Label",
    "timestamp": "TimeStamp", 
    "cpuid": "CPUID",
    "syscall": "SysCall",
}

if __name__ == "__main__":
    
    log_path = "/CAShift/Dataset/data_process/test"

    ########################
    ### For train normal ###
    ########################
    normal_path = f"{log_path}/normal/all.scap"
    normal_df = get_scap_dfs([normal_path], "normal", -1)

    chunk_size = 10000
    log_number = len(normal_df)//chunk_size

    print(f"total {log_number} logs")
    
    chunks = [normal_df[i*chunk_size:(i+1)*chunk_size] for i in range(log_number)]
    
    # create output directory
    os.makedirs(f"{log_path}/output/normal", exist_ok=True)
    
    # save to csv
    for i, chunk in enumerate(chunks):
        chunk = chunk.reset_index(drop=True)
        chunk.rename_axis("LineId", inplace=True)
        chunk.rename(columns=columns, inplace=True)
        # normal_df["EventId"] = normal_df.groupby("Syscall").ngroup()
        log_frame = chunk[["Label", "TimeStamp", "CPUID", "SysCall"]]
        chunk.to_csv(f"{log_path}/output/normal/normal_{i}.csv", index=False)
    print(f"train normal saved")
    
    ########################
    ### For test normal ###
    ########################
    test_path = f"{log_path}/test/all.scap"
    test_df = get_scap_dfs([test_path], "normal", -1)
    
    # create output directory
    os.makedirs(f"{log_path}/output/test", exist_ok=True)
    
    total_size = len(test_df)
    record_size = 0
    log_number = 0

    while True:
        chunk_size = random.randint(5000, 15000)
        if total_size > record_size + chunk_size:
            chunk = test_df[record_size:record_size+chunk_size]
            chunk = chunk.reset_index(drop=True)
            
            chunk.rename_axis("LineId", inplace=True)
            chunk.rename(columns=columns, inplace=True)
            # normal_df["EventId"] = normal_df.groupby("Syscall").ngroup()
            log_frame = chunk[["Label", "TimeStamp", "CPUID", "SysCall"]]
            chunk.to_csv(f"{log_path}/output/test/normal_{log_number}.csv", index=False)
            record_size += chunk_size
            log_number += 1
        else:
            break

    print(f"total {log_number} logs")
    print(f"test normal saved")
    
    #################################
    ### For collected test attack ###
    #################################
    attack_path = f"{log_path}/attack/"
    attack_id = "CVE-2019-5736" ########### Remember to Change
    # create output directory
    os.makedirs(f"{log_path}/output/attack", exist_ok=True)

    # for every scap file in attack_path
    for i, filename in enumerate(os.listdir(attack_path)):
        attack_scap_path = os.path.join(attack_path, filename)
        attack_df = get_scap_dfs([attack_scap_path], attack_id, i)
        attack_df.rename_axis("LineId", inplace=True)
        attack_df.rename(columns=columns, inplace=True)
    
        log_frame = attack_df[["Label", "TimeStamp", "CPUID", "SysCall"]]
        attack_df.to_csv(f"{log_path}/output/attack/{attack_id}_{i}.csv", index=False)

    print(f"total {i+1} logs")
    print(f"test attack saved")

    ### For CB-DS dataset attacks ###
    # attack_path = f"{log_path}/attack/all.scap"
    # attack_df = get_scap_dfs([attack_path], True)
    # attack_id = "CVE-2016-10033"
    
    # # create output directory
    # os.makedirs(f"{log_path}/output/attack", exist_ok=True)

    # total_size = len(attack_df)
    # record_size = 0
    # log_number = 0

    # while True:
    #     chunk_size = random.randint(5000, 15000)
    #     if total_size > record_size + chunk_size:
    #         chunk = attack_df[record_size:record_size+chunk_size]
    #         chunk = chunk.reset_index(drop=True)
            
    #         chunk.rename_axis("LineId", inplace=True)
    #         chunk.rename(columns=columns, inplace=True)
    #         # normal_df["EventId"] = normal_df.groupby("Syscall").ngroup()
    #         log_frame = chunk[["TimeStamp", "CPUID", "SysCall"]]
    #         chunk.to_csv(f"{log_path}/output/attack/{attack_id}_{log_number}.csv", index=False)
    #         record_size += chunk_size
    #         log_number += 1
    #     else:
    #         break

    # print(f"total {log_number} logs")
    # print(f"test attack saved")