import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from azure.storage.blob import BlobProperties, ContainerClient
from tqdm import tqdm

DIRECTORY_PREFIX = {
    "meta": "azureml/d31f4502-53eb-4635-b40b-50dc617cfe0b/merged_preprocessed_path/",
    "files": "azureml/d31f4502-53eb-4635-b40b-50dc617cfe0b/ecg_files_path/",
}


def get_blob_list(container_client: ContainerClient, directory_prefix: str):
    return container_client.list_blobs(name_starts_with=directory_prefix)


def download_blob(blob: BlobProperties, container_client, target_dir, subdirs=False):
    if subdirs:
        target_file = Path(target_dir) / Path(blob.name)
    else:
        target_file = Path(target_dir) / Path(blob.name).name

    with open(target_file, mode="wb") as download_file:
        download_file.write(container_client.download_blob(blob.name).readall())

    return target_file


def download_blobs_in_directory(
    sas_url: str, directory_prefix: str, target_dir: str, subdirs=False, max_workers=8
) -> None:

    container_client = ContainerClient.from_container_url(sas_url)

    print("Listing blobs...")
    blob_list = get_blob_list(container_client, directory_prefix)

    print(f"Downloading blobs...")

    _download_blob = partial(
        download_blob,
        container_client=container_client,
        target_dir=target_dir,
        subdirs=subdirs,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_blob, blob) for blob in blob_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


def main():
    parser = argparse.ArgumentParser(
        description="Download blobs from Azure Blob Storage."
    )
    parser.add_argument(
        "--directory_prefix",
        type=str,
        required=True,
        help="The prefix of the directory to list blobs from. Meta or files",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="The directory to download the blobs to.",
    )

    args = parser.parse_args()

    assert args.directory_prefix in DIRECTORY_PREFIX, "Invalid directory prefix."

    sas_url = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    if sas_url is None:
        raise ValueError(
            "Environment variable AZURE_STORAGE_CONNECTION_STRING is not set. Do:\nexport AZURE_STORAGE_CONNECTION_STRING='your_sas_url_here'"
        )

    download_blobs_in_directory(
        sas_url, DIRECTORY_PREFIX[args.directory_prefix], args.target_dir, False
    )


if __name__ == "__main__":
    main()
