
from pathlib import Path
import pandas as pd

# CSV file ka correct path
file_path = Path(
    "data/raw/m5-forecasting-accuracy/sales_train_validation.csv"
)

# Check karo ki file exist karti hai
if not file_path.exists():
    print("CSV file nahi mili. File path check karo.")
else:
    print("CSV file mil gayi!")

    # Data ko read karo
    df = pd.read_csv(file_path)

    # Basic information
    print("\n1. Dataset ka size:")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n2. Pehli 5 rows:")
    print(df.head())

    print("\n3. Column names:")
    print(df.columns.tolist()[:10])

    print("\n4. Missing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\n5. Dataset information:")
    df.info()
