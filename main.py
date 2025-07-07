from pipeline.runner import run_pipeline
import pickle as pkl
import pandas as pd

if __name__ == '__main__':
    stats_df = run_pipeline()
    print(f"\n📦 Loaded stats_df with shape: {stats_df.shape}")
    print("📄 Columns:", stats_df.columns.tolist())
    print("\n🔍 Preview of stats_df:")
    print(stats_df.head(10))
    with open("./output/DAILY_SUMMARIES/stats_tensor.pkl", "rb") as f:
        object = pkl.load(f)
    df = pd.DataFrame(object)
    df.to_csv(r'./output/DAILY_SUMMARIES/stats_tensor.csv')
