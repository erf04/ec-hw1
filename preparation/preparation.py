import pandas as pd

def load_data(file_path):
    """
    Loads a sales dataset from a CSV or Excel file.
    
    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    


def clean_data(df):
    """
    Cleans the sales dataset by removing nulls, duplicates, and invalid values.
    """
    # 1. Drop duplicate rows
    df = df.drop_duplicates()

    # 2. Remove rows with null values in key columns
    df = df.dropna(subset=['Description', 'Price', 'Quantity'])

    # 3. Remove negative or zero prices and quantities
    df = df[(df['Price'] > 0) & (df['Quantity'] > 0)]

    # 4. Standardize text columns (optional)
    if 'Description' in df.columns:
        df['Description'] = df['Description'].str.strip().str.lower()

    print(f"✅ Data cleaned! Remaining rows: {len(df)}")
    return df


def summarize_data(df):
    """
    Summarizes sales data by product: average price, total quantity sold, and number of unique customers.
    """
    summary = (
        df.groupby('Description')
          .agg({
              'Price': 'mean',
              'Quantity': 'sum',
              'Customer ID': pd.Series.nunique
          })
          .reset_index()
          .rename(columns={
              'Price': 'AvgPrice',
              'Quantity': 'TotalQuantity',
              'Customer ID': 'UniqueCustomers'
          })
    )

    print(f"✅ Summary created! {len(summary)} unique products.")
    return summary


