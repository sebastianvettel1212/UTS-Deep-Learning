import gzip
import csv
from collections import defaultdict

def analyze_rul(filepath):
    print(f"=== Analyzing {filepath} ===")
    units = defaultdict(list)
    missing_counts = defaultdict(int)
    total_rows = 0
    with gzip.open(filepath, 'rt') as f:
        reader = csv.reader(f)
        header = next(reader)
        sensor_cols = header[5:]
        for row in reader:
            total_rows += 1
            uid = int(row[0])
            cycle = int(row[1])
            units[uid].append(cycle)
            for col_idx, col_name in enumerate(sensor_cols, start=5):
                val = row[col_idx]
                if val == '' or val.lower() == 'nan':
                    missing_counts[col_name] += 1
    
    print(f"Total rows: {total_rows}")
    print(f"Total unique units: {len(units)}")
    cycle_lengths = [len(cycles) for cycles in units.values()]
    print(f"Cycle lengths: min={min(cycle_lengths)}, max={max(cycle_lengths)}, avg={sum(cycle_lengths)/len(cycle_lengths):.2f}")
    print("Missing sensor values (%):")
    for col in sensor_cols:
        pct = (missing_counts[col] / total_rows) * 100
        if pct > 0:
            print(f"  {col}: {missing_counts[col]} ({pct:.2f}%)")

analyze_rul('data/rul_train.csv.gz')
print()
analyze_rul('data/rul_test.csv.gz')
