# get_data.py

import kagglehub
import os
import shutil

def download_data_with_kagglehub():
    """
    Downloads the UNSW-NB15 dataset from user 'dhoogla' and
    copies the main CSV file to the current directory.
    """
    # The new CSV file we need from this specific dataset
    required_csv_file = 'UNSW_NB15_training-set.csv'

    if os.path.exists(required_csv_file):
        print(f"'{required_csv_file}' already exists. Skipping download.")
        return

    print("Downloading dataset using kagglehub...")
    try:
        # The new Kaggle dataset handle
        handle = 'dhoogla/unswnb15/unsw-nb15/1'

        dataset_path = kagglehub.dataset_download(handle)
        
        print(f"Dataset downloaded to cache: {dataset_path}")

        source_csv_path = os.path.join(dataset_path, required_csv_file)

        if os.path.exists(source_csv_path):
            shutil.copy(source_csv_path, '.')
            print(f"Successfully copied '{required_csv_file}' to project folder.")
        else:
            print(f"Error: Could not find '{required_csv_file}' in the downloaded dataset.")
            print("Available files:", os.listdir(dataset_path))

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nPlease ensure your kaggle.json token is set up correctly.")
        print("IMPORTANT: Make sure you have accepted the dataset's rules on the Kaggle website first.")

if __name__ == "__main__":
    download_data_with_kagglehub()