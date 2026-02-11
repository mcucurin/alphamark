import pandas as pd

# Read CSV
df = pd.read_csv("/Users/dhruvpatel/financial_pipeline/test_input/features_20250102.csv")

# Save as pickle
df.to_pickle("features_20250102.pkl")
