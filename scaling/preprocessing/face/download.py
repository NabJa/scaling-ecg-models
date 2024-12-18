import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from azure.storage.blob import BlobProperties, BlobServiceClient, ContainerClient
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

    service_client = BlobServiceClient(account_url=sas_url)
    container_client = service_client.get_container_client(
        "azureml-blobstore-7209d09f-8ee3-41a4-9ebc-7034bca04b1c"
    )

    blob_list = get_blob_list(container_client, directory_prefix)

    _download_blob = partial(
        download_blob,
        container_client=container_client,
        target_dir=target_dir,
        subdirs=subdirs,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print("Listing blobs...")
        futures = [executor.submit(_download_blob, blob) for blob in blob_list]

        print(f"Downloading blobs...")
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

    assert (
        args.directory_prefix in DIRECTORY_PREFIX
    ), "Invalid directory prefix. Must be 'meta' or 'files'."

    sas_url = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

    if sas_url is None:
        raise ValueError(
            "Azure Storage Connection String not found. Set as AZURE_STORAGE_CONNECTION_STRING environment variable."
        )

    download_blobs_in_directory(
        sas_url, DIRECTORY_PREFIX[args.directory_prefix], args.target_dir, False
    )


if __name__ == "__main__":
    main()
