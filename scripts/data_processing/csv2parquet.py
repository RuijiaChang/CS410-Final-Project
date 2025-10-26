import pandas as pd

df = pd.read_csv("data/processed/interaction.csv")

print(df.dtypes)

df.to_parquet("data/processed/interactions.parquet", index=False)

print("Saved as data/processed/interactions.parquet")
