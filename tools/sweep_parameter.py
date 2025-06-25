import csv
import itertools
import copy

def sweep_csv(input_csv_path, output_csv_path, sweep_config):
    """
    Perform a parameter sweep on specified columns of a CSV file.
    
    For each row in the base CSV, generate combinations only on the sweep_config fields.
    Correlated fields like Device/TensorParallelSize are left unchanged.
    Only combinations where MaxNumBatchedTokens >= MaxNumSeqs are included.
    
    Parameters:
        input_csv_path (str): Path to base CSV file.
        output_csv_path (str): Path to save sweep output CSV.
        sweep_config (dict): Keys are column names to sweep, values are lists of values.
    """
    with open(input_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)
        header = reader.fieldnames

    # Validate sweep keys
    for col in sweep_config:
        if col not in header:
            raise ValueError(f"Column '{col}' not found in CSV header.")

    # Generate output rows
    expanded_rows = []
    for base_row in base_rows:
        sweep_values = list(itertools.product(*(sweep_config[col] for col in sweep_config)))
        for values in sweep_values:
            sweep_dict = dict(zip(sweep_config.keys(), values))
            max_seqs = int(sweep_dict.get("MaxNumSeqs", 0))
            max_tokens = int(sweep_dict.get("MaxNumBatchedTokens", 0))
            if "MaxNumSeqs" in sweep_dict and "MaxNumBatchedTokens" in sweep_dict:
                if max_tokens < max_seqs:
                    continue  # Skip invalid combination
            new_row = copy.deepcopy(base_row)
            for col, val in sweep_dict.items():
                new_row[col] = str(val)
            expanded_rows.append(new_row)

    # Write output CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(expanded_rows)

    print(f"Sweep complete: {len(expanded_rows)} rows written to {output_csv_path}")


sweep_csv(
    input_csv_path="/home/cuiq_google_com/base.csv",
    output_csv_path="/home/cuiq_google_com/tune_ali_qwen.csv",
    sweep_config={
        "MaxNumSeqs": [64, 128, 256, 512],
        "MaxNumBatchedTokens": [64, 128, 256, 512, 1024, 2048, 4096]
    }
)