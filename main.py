import os
from pathlib import Path
from utils.get_Volume import get_excel_volume_score
import pandas as pd
import random
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

def analyze_single_excel_file(directory_path: str):
    """
    Randomly select and analyze one Excel file from the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing Excel files
    """
    # Convert to Path object
    dir_path = Path(directory_path)
    
    # Get all Excel files
    excel_files = list(dir_path.glob('*.xlsx'))
    
    if not excel_files:
        print(f"No Excel files found in {directory_path}")
        return
    
    # Randomly select one file
    selected_file = random.choice(excel_files)
    print(f"\nSelected file: {selected_file.name}")
    
    try:
        # Process the selected file
        result = get_excel_volume_score(selected_file)
        print(f"Size: {result['size_gb']:.4f} GB")
        print(f"Volume Score: {result['volume_score']:.4f}")
        print(f"Rows: {result['file_info'].get('rows', 'N/A')}")
        print(f"Columns: {result['file_info'].get('columns', 'N/A')}")
        
    except Exception as e:
        print(f"Error processing {selected_file.name}: {str(e)}")

def analyze_all_excel_files(directory_path: str):
    """
    Analyze all Excel files in the specified directory and show their volume scores.
    
    Args:
        directory_path (str): Path to the directory containing Excel files
    """
    # Convert to Path object
    dir_path = Path(directory_path)
    
    # Get all Excel files
    excel_files = list(dir_path.glob('*.xlsx'))
    
    if not excel_files:
        print(f"No Excel files found in {directory_path}")
        return
    
    # Create a list to store results
    results = []
    
    # Process each Excel file
    for file_path in excel_files:
        try:
            result = get_excel_volume_score(file_path)
            results.append({
                'file_name': file_path.name,
                'size_gb': result['size_gb'],
                'volume_score': result['volume_score'],
                'rows': result['file_info'].get('rows', 'N/A'),
                'columns': result['file_info'].get('columns', 'N/A')
            })
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    # Convert results to DataFrame for better display
    df_results = pd.DataFrame(results)
    
    # Sort by volume score in descending order
    df_results = df_results.sort_values('volume_score', ascending=False)
    
    # Format the size column to 4 decimal places
    df_results['size_gb'] = df_results['size_gb'].map('{:.4f}'.format)
    
    # Display results
    print("\n=== Analysis Results for All Files ===")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    # Path to the directory containing Excel files
    excel_dir = r"e:\wxq_postgradute\数据集\extracted_excel_files"
    
    # Get user choice
    print("请选择分析模式：")
    print("1. 随机分析一个文件")
    print("2. 分析所有文件")
    
    choice = input("请输入选项（1或2）：").strip()
    
    if choice == "1":
        analyze_single_excel_file(excel_dir)
    elif choice == "2":
        analyze_all_excel_files(excel_dir)
    else:
        print("无效的选项，请输入1或2")
