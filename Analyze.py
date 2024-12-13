import json
import pandas as pd
from typing import Tuple

def convert_json_to_csv(json_file_path: str, instance_index: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert JSON problem instance to stock.csv and demand.csv format
    
    Args:
        json_file_path: Path to the JSON file
        instance_index: Index of the instance to convert (default 0 for first instance)
        
    Returns:
        Tuple of (stocks_df, demands_df)
    """
    # Read JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get the specified instance
    instance = data['instances'][instance_index]
    
    # Create stock dataframe
    stocks_data = []
    for i, stock in enumerate(instance['stocks']):
        stocks_data.append({
            'id': i,
            'length': stock['length'],
            'width': stock['width']
        })
    stocks_df = pd.DataFrame(stocks_data)
    
    # Create demand dataframe
    demands_data = []
    for i, item in enumerate(instance['items']):
        demands_data.append({
            'id': i,
            'length': item['length'],
            'width': item['width'],
            'quantity': item['demand']
        })
    demands_df = pd.DataFrame(demands_data)
    
    return stocks_df, demands_df

def save_to_csv(json_file_path: str, output_prefix: str = 'converted', instance_index: int = 0):
    """
    Save the converted dataframes to CSV files
    """
    stocks_df, demands_df = convert_json_to_csv(json_file_path, instance_index)
    
    # Save to CSV
    stock_file = f'{output_prefix}_stock.csv'
    demand_file = f'{output_prefix}_demand.csv'
    
    stocks_df.to_csv(stock_file, index=False)
    demands_df.to_csv(demand_file, index=False)
    
    print(f"Converted files saved as {stock_file} and {demand_file}")
    print("\nStock data preview:")
    print(stocks_df.head())
    print("\nDemand data preview:")
    print(demands_df.head())

if __name__ == "__main__":
    # Example usage
    json_file = "Datasets/problem_10.json"
    save_to_csv(json_file)