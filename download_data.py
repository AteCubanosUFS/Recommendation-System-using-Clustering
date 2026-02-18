import os
import shutil
import zipfile
import requests
from tqdm import tqdm

URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "ml-32m.zip")
EXTRACTED_FOLDER = os.path.join(DATA_DIR, "ml-32m")

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as file, tqdm(
        desc="Download",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.infolist()
        total_size = sum(file.file_size for file in file_list)

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Extração"
        ) as bar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                bar.update(file.file_size)

def move_csv_files(source_folder, destination_folder):
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".csv"):
            shutil.move(
                os.path.join(source_folder, file_name),
                os.path.join(destination_folder, file_name)
            )

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        download_file(URL, ZIP_PATH)

    if not os.path.exists(EXTRACTED_FOLDER):
        extract_zip(ZIP_PATH, DATA_DIR)

    if os.path.exists(EXTRACTED_FOLDER):
        move_csv_files(EXTRACTED_FOLDER, DATA_DIR)
        shutil.rmtree(EXTRACTED_FOLDER)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print("Processo finalizado com sucesso")

if __name__ == "__main__":
    main()