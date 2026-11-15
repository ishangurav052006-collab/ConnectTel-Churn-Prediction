"""Download the IBM Telco Customer Churn dataset."""

import os
import sys
import urllib.request
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Telco-Customer-Churn.csv")


def download_data(url: str = DATA_URL, output_path: str = OUTPUT_PATH) -> str:
    """Download dataset from the specified URL and save to output_path.

    Args:
        url: Direct CSV download URL.
        output_path: Target file path on local filesystem.

    Returns:
        The target file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Downloading dataset from: {url}")
    logging.info(f"Target destination: {output_path}")

    # Use a custom user agent to avoid bot filtering
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    file_size = os.path.getsize(output_path)
    logging.info(f"Successfully downloaded {file_size:,} bytes to {output_path}")
    return output_path


if __name__ == "__main__":
    download_data()

