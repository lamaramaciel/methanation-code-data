import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.
    
    Args:
        filepath (str): Path to the Excel file.
        
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Load the dataset
    data = pd.read_excel(filepath, sheet_name='PlanML')
    
    # Drop rows with missing values
    data.dropna(axis=0, inplace=True)
    
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

# Example usage
if __name__ == "__main__":
    # Adjust the path to your dataset file
    dataset_path = 'data/data_art1.xlsx'
    
    # Load and preprocess the data
    data = load_and_preprocess_data(dataset_path)
    
    # Display the first few rows
    print(data.head())

