import os
import urllib.request
import zipfile

DATASET_ZIP_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
TARGET_DIR = "data/raw/ptbxl"


def download_and_extract_ptbxl():
    """
    Downloads and extracts PTB-XL dataset (v1.0.3) from PhysioNet directly into data/raw/ptbxl/.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)
    db_csv = os.path.join(TARGET_DIR, "ptbxl_database.csv")

    if os.path.exists(db_csv):
        print(f"PTB-XL dataset already exists in '{TARGET_DIR}'. Ready to go!")
        return

    zip_path = "data/raw/ptbxl_dataset.zip"
    print(f"Downloading PTB-XL dataset from PhysioNet ({DATASET_ZIP_URL})...")
    print("This file is approximately 3.0 GB. Please wait...")

    try:
        urllib.request.urlretrieve(DATASET_ZIP_URL, zip_path)
        print(f"Download finished! Extracting to '{TARGET_DIR}'...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data/raw")

        # Clean up nested folders if zip creates a wrapper folder
        extracted_folder = os.path.join("data/raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
        if os.path.exists(extracted_folder):
            for item in os.listdir(extracted_folder):
                s = os.path.join(extracted_folder, item)
                d = os.path.join(TARGET_DIR, item)
                if not os.path.exists(d):
                    os.rename(s, d)
            os.rmdir(extracted_folder)

        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"PTB-XL Dataset successfully setup in '{TARGET_DIR}'!")
    except Exception as e:
        print(f"Error downloading dataset automatically: {e}")
        print("You can manually download PTB-XL v1.0.3 from PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/")
        print(f"Extract all files into '{TARGET_DIR}/'.")


if __name__ == "__main__":
    download_and_extract_ptbxl()
