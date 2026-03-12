import pandas as pd

# original wafer dataset
df = pd.read_csv("prediction_artifacts/wafer_23012020_041211.csv")

# target column remove
df = df.drop(columns=["Good/Bad"])

# sirf 5 rows test layi


# new csv for prediction
df.to_csv("test.csv", index=False)

print("test.csv created successfully")