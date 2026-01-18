"""
Dataset Pre-download Script

Download all required datasets for the baseline evaluation to avoid
network issues during experiments.

Usage:
    python download_datasets.py
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "splits": ["test"],
        "trust_remote_code": False
    },
    "c4": {
        "path": "allenai/c4",
        "name": None,
        "data_files": {"validation": "en/c4-validation.00000-of-00008.json.gz"},
        "splits": ["validation"],
        "trust_remote_code": False
    },
    "ptb": {
        "path": "ptb_text_only",
        "name": "penn_treebank",
        "splits": ["test"],
        "trust_remote_code": True
    },
    "hellaswag": {
        "path": "hellaswag",
        "name": None,
        "splits": ["validation"],
        "trust_remote_code": False
    },
    "piqa": {
        "path": "piqa",
        "name": None,
        "splits": ["validation"],
        "trust_remote_code": False
    },
    "arc_easy": {
        "path": "ai2_arc",
        "name": "ARC-Easy",
        "splits": ["test"],
        "trust_remote_code": False
    },
    "boolq": {
        "path": "super_glue",
        "name": "boolq",
        "splits": ["validation"],
        "trust_remote_code": False
    },
    "pileval": {
        "path": "mit-han-lab/pile-val-backup",
        "name": None,
        "splits": ["validation"],
        "trust_remote_code": False
    }
}


def download_dataset(dataset_name: str, config: dict, max_retries: int = 5):
    """Download a single dataset with retry logic"""
    logger.info(f"Downloading {dataset_name}...")
    
    for attempt in range(max_retries):
        try:
            load_kwargs = {
                "path": config["path"],
                "trust_remote_code": config["trust_remote_code"]
            }
            
            if config["name"]:
                load_kwargs["name"] = config["name"]
            
            if "data_files" in config:
                load_kwargs["data_files"] = config["data_files"]
            
            # Try to load each split
            for split in config["splits"]:
                logger.info(f"  Loading split: {split}")
                dataset = load_dataset(**load_kwargs, split=split)
                logger.info(f"  ✓ {split}: {len(dataset)} samples")
            
            logger.info(f"✓ Successfully downloaded {dataset_name}\n")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                logger.warning(f"  Retrying in {2 ** attempt}s...")
                import time
                time.sleep(2 ** attempt)
            else:
                logger.error(f"✗ Failed to download {dataset_name} after {max_retries} attempts")
                logger.error(f"  Error: {e}\n")
                return False


def check_existing_datasets():
    """Check which datasets are already cached"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    
    logger.info("Checking existing datasets in cache...")
    logger.info(f"Cache directory: {cache_dir}\n")
    
    if not cache_dir.exists():
        logger.info("No cache directory found. All datasets need to be downloaded.\n")
        return set()
    
    existing = set()
    for dataset_name, config in DATASETS.items():
        dataset_path = config["path"].replace("/", "___")
        if config["name"]:
            dataset_path += f"___{config['name']}"
        
        # Check if any matching directory exists
        matching_dirs = list(cache_dir.glob(f"{dataset_path}*"))
        if matching_dirs:
            existing.add(dataset_name)
            logger.info(f"✓ {dataset_name}: Found in cache")
        else:
            logger.info(f"✗ {dataset_name}: Not in cache")
    
    logger.info("")
    return existing


def main():
    """Main download function"""
    logger.info("=" * 80)
    logger.info("DATASET PRE-DOWNLOAD SCRIPT")
    logger.info("=" * 80)
    logger.info("")
    
    # Check existing datasets
    existing = check_existing_datasets()
    
    # Determine what needs to be downloaded
    to_download = [name for name in DATASETS.keys() if name not in existing]
    
    if not to_download:
        logger.info("All datasets are already cached!")
        logger.info("You can run baseline.py directly.")
        return
    
    logger.info(f"Need to download {len(to_download)} datasets:")
    for name in to_download:
        logger.info(f"  - {name}")
    logger.info("")
    
    # Download missing datasets
    logger.info("=" * 80)
    logger.info("STARTING DOWNLOADS")
    logger.info("=" * 80)
    logger.info("")
    
    success_count = 0
    failed_datasets = []
    
    for dataset_name in tqdm(to_download, desc="Overall Progress"):
        config = DATASETS[dataset_name]
        success = download_dataset(dataset_name, config)
        
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
    
    # Summary
    logger.info("=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successfully downloaded: {success_count}/{len(to_download)}")
    
    if failed_datasets:
        logger.info(f"\nFailed datasets:")
        for name in failed_datasets:
            logger.info(f"  ✗ {name}")
        logger.info("\nYou can try downloading these manually or run this script again.")
        logger.info("Alternatively, you can skip failed datasets in baseline.py config.")
    else:
        logger.info("\n✓ All datasets downloaded successfully!")
        logger.info("You can now run baseline.py with confidence.")
    
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)
