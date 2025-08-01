from pipeline.loader import load_daily_files
from pipeline.features import generate_signals_and_targets

# Load data
daily_data = load_daily_files('data/RAW_DATA')
print(f"Loaded {len(daily_data)} daily files")

# Check first few files for 'open' column
print("\nChecking for 'open' column in first few files:")
for i, (fname, df) in enumerate(daily_data[:3]):
    has_open = 'open' in df.columns
    print(f"{fname}: has 'open' column = {has_open}")

# Generate signals
enriched_data = generate_signals_and_targets(daily_data)

# Check what signals were generated
print("\nChecking signals in first few enriched files:")
for i, (fname, df) in enumerate(enriched_data[:3]):
    signal_cols = [col for col in df.columns if col.startswith('pret_')]
    print(f"{fname}: signal columns = {signal_cols}")

# Debug the open-to-close signal generation
print("\nDebugging open-to-close signal generation:")
for i in range(1, min(3, len(daily_data))):  # Start from 1 to skip first file
    fname, df = daily_data[i]
    prev_df = daily_data[i-1][1]
    print(f"\nFile {i}: {fname}")
    print(f"  Previous file has 'open': {'open' in prev_df.columns}")
    print(f"  Current file has 'prevAdjClose': {'prevAdjClose' in df.columns}")
    
    # Simulate the merge
    merged = df[['ticker', 'prevAdjClose']].merge(
        prev_df[['ticker', 'open']], on='ticker', suffixes=('', '_past1'))
    print(f"  Merged columns: {list(merged.columns)}")
    print(f"  Has 'open_past1': {'open_past1' in merged.columns}") 