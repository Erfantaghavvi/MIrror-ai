import pandas as pd
import sys

def sort_csv(input_file, output_file, sort_column=None):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get list of available columns
    columns = df.columns.tolist()
    
    if sort_column is None:
        # Print available columns for sorting
        print("\nAvailable columns for sorting:")
        for i, col in enumerate(columns):
            print(f"{i+1}. {col}")
        
        # Get user input for sorting column
        try:
            choice = int(input("\nEnter the number of the column you want to sort by: "))
            if 1 <= choice <= len(columns):
                sort_column = columns[choice-1]
            else:
                print("Invalid column number. Using 'Product_Name' as default.")
                sort_column = 'Product_Name'
        except ValueError:
            print("Invalid input. Using 'Product_Name' as default.")
            sort_column = 'Product_Name'
    
    # Sort the dataframe
    df_sorted = df.sort_values(by=sort_column)
    
    # Save the sorted data to a new CSV file
    df_sorted.to_csv(output_file, index=False)
    print(f"\nFile has been sorted by '{sort_column}' and saved as '{output_file}'")

if __name__ == "__main__":
    input_file = "most_used_beauty_cosmetics_products_extended.csv"
    output_file = "sorted_cosmetics_products.csv"
    sort_csv(input_file, output_file)
