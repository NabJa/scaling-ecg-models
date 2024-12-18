import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_local_path(base, url) -> Path:
    """Extract the path from the URL and create the local path."""

    # Extract the path from the URL and create the local path
    # e.g. .../training/chapman_shaoxing/g1/JS00001.hea?download' -> chapman_shaoxing/g1/JS00001.hea
    url_path = Path(*url.split("?")[0].split("/")[-3:])

    local_path = Path(base) / url_path
    local_path.parent.mkdir(exist_ok=True, parents=True)
    return local_path


def download_file(session, url, local_filename) -> Optional[str]:
    """Download a file from a URL to a local file if it doesn't already exist."""

    if local_filename.exists():
        return local_filename

    try:
        with session.get(url, stream=True, timeout=(120, 120)) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None


def generate_dataset_urls() -> List[str]:
    """Generate a list of URLs for all datasets."""
    base_url = "https://physionet.org/content/challenge-2021/1.0.3/training/"
    datasets = [
        "chapman_shaoxing",
        "cpsc_2018",
        "cpsc_2018_extra",
        "georgia",
        "ningbo",
        "ptb",
        "ptb-xl",
        "st_petersburg_incart",
    ]
    urls = []
    for dataset in datasets:
        for i in range(1, 36):
            url = urljoin(base_url, f"{dataset}/g{i}/#files-panel")
            urls.append(url)
    return urls


def get_file_urls(session, base_url) -> List[str]:
    """From a dataset URL, return a list of all file URLs."""
    try:
        response = session.get(base_url)
        soup = BeautifulSoup(response.text, "html.parser")
        return [
            urljoin(base_url, a["href"])
            for a in soup.find_all("a", attrs={"class": "download"})
            if not a["href"].endswith("/")
        ]
    except Exception as e:
        print(f"Error fetching file URLs from {base_url}: {str(e)}")
        return []


def get_all_files(dataset_urls: List[str], max_workers: int = 10) -> List[str]:
    """From a list of dataset URLs, return a list of all file URLs."""
    all_file_urls = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with requests.Session() as session:
            file_url_futures = {
                executor.submit(get_file_urls, session, url): url
                for url in dataset_urls
            }
            for future in tqdm(
                as_completed(file_url_futures),
                total=len(dataset_urls),
                desc="Fetching file URLs",
            ):
                all_file_urls.extend(future.result())

    return all_file_urls


def main():
    max_workers = 10
    local_base_dir = (
        "/sc-scratch/sc-scratch-gbm-radiomics/ecg/physionet_challenge/training"
    )
    os.makedirs(local_base_dir, exist_ok=True)

    dataset_urls = generate_dataset_urls()
    all_file_urls = get_all_files(dataset_urls, max_workers=10)

    total_files = len(all_file_urls)

    start_time = time.time()
    files_downloaded = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with requests.Session() as session:
            future_to_url = {
                executor.submit(
                    download_file, session, url, get_local_path(local_base_dir, url)
                ): url
                for url in all_file_urls
            }

            for future in tqdm(
                as_completed(future_to_url), total=total_files, desc="Downloading files"
            ):
                url = future_to_url[future]
                try:
                    future_result = future.result()
                    if future_result:
                        files_downloaded += 1
                except Exception as exc:
                    print(f"{url} generated an exception: {exc}")

    total_time = time.time() - start_time
    print(f"Download completed. Total files: {files_downloaded}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {files_downloaded / total_time:.2f} files/second")


if __name__ == "__main__":
    main()
