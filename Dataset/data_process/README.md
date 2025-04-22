# Process Dataset into Small Instances

## For LogADEmpirical

Gather all logs into one single csv file containing `log_frame[["Label", "Timestamp", "Content", "EventId", "EventTemplate", "ParameterList"]]

1. parser.py 
2. generate_embedding.py

**Procedures**
1. rm -rf test/output/*
2. get all.scap: cat *.scap > ../all.scap
3. change log_path, attack_id in parser_dataset.py
4. python parse_dataset.py 

For normal logs, use convert_files.py
1. convert_files.py // change label to "normal_{index}"
2. combine_all_files.py // merge all seperate files into one csv for DeepLog baseline
