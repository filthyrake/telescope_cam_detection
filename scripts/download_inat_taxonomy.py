#!/usr/bin/env python3
"""
Download and extract iNaturalist 2021 taxonomy mapping.
Creates a JSON file mapping class IDs to species names.
"""

import json
import logging
import urllib.request
import tarfile
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL for iNaturalist 2021 validation dataset (smaller than train)
VAL_JSON_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz"

def download_and_extract_taxonomy(output_dir: str = "models"):
    """
    Download iNaturalist 2021 dataset and extract taxonomy.

    Args:
        output_dir: Directory to save taxonomy files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    taxonomy_file = output_path / "inat2021_taxonomy.json"

    if taxonomy_file.exists():
        logger.info(f"Taxonomy file already exists: {taxonomy_file}")
        # Load and display count
        with open(taxonomy_file, 'r') as f:
            taxonomy = json.load(f)
            logger.info(f"Found {len(taxonomy)} species in taxonomy")
        return str(taxonomy_file)

    logger.info("Downloading iNaturalist 2021 validation dataset...")
    logger.info(f"URL: {VAL_JSON_URL}")

    try:
        # Download to temporary file
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name

            logger.info("Downloading (this may take a few minutes)...")
            urllib.request.urlretrieve(VAL_JSON_URL, tmp_path)
            logger.info(f"Downloaded to {tmp_path}")

            # Extract JSON from tar.gz
            logger.info("Extracting taxonomy data...")
            with tarfile.open(tmp_path, 'r:gz') as tar:
                # Extract val.json to memory
                json_member = tar.getmember('val.json')
                json_file = tar.extractfile(json_member)

                # Parse JSON
                logger.info("Parsing dataset JSON...")
                data = json.load(json_file)

                # Extract categories (species information)
                categories = data.get('categories', [])
                logger.info(f"Found {len(categories)} species categories")

                # Create taxonomy mapping: class_id -> species info
                taxonomy = {}
                for category in categories:
                    class_id = category['id']

                    # Create a comprehensive species entry
                    taxonomy[str(class_id)] = {
                        'name': category['name'],
                        'common_name': category.get('common_name', ''),
                        'kingdom': category.get('kingdom', ''),
                        'phylum': category.get('phylum', ''),
                        'class': category.get('class', ''),
                        'order': category.get('order', ''),
                        'family': category.get('family', ''),
                        'genus': category.get('genus', ''),
                        'specific_epithet': category.get('specific_epithet', '')
                    }

                # Save taxonomy file
                logger.info(f"Saving taxonomy to {taxonomy_file}...")
                with open(taxonomy_file, 'w') as f:
                    json.dump(taxonomy, f, indent=2)

                logger.info(f"✅ Taxonomy saved successfully!")
                logger.info(f"   File: {taxonomy_file}")
                logger.info(f"   Species count: {len(taxonomy)}")

                # Also create a simplified version (just class_id -> common name)
                simple_taxonomy_file = output_path / "inat2021_taxonomy_simple.json"
                simple_taxonomy = {}
                for class_id, info in taxonomy.items():
                    # Prefer common name, fall back to scientific name
                    display_name = info['common_name'] if info['common_name'] else info['name']
                    simple_taxonomy[class_id] = display_name

                with open(simple_taxonomy_file, 'w') as f:
                    json.dump(simple_taxonomy, f, indent=2)

                logger.info(f"✅ Simplified taxonomy saved: {simple_taxonomy_file}")

                # Print some example species
                logger.info("\nExample species (first 10):")
                for i, (class_id, info) in enumerate(list(taxonomy.items())[:10]):
                    name = info['common_name'] if info['common_name'] else info['name']
                    logger.info(f"  {class_id}: {name} ({info['family']})")

                # Clean up temporary file
                Path(tmp_path).unlink()

                return str(taxonomy_file)

    except Exception as e:
        logger.error(f"Failed to download taxonomy: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logger.info("iNaturalist 2021 Taxonomy Downloader")
    logger.info("=" * 80)

    taxonomy_path = download_and_extract_taxonomy()

    if taxonomy_path:
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS! Taxonomy files ready for use.")
        logger.info(f"Full taxonomy: models/inat2021_taxonomy.json")
        logger.info(f"Simple taxonomy: models/inat2021_taxonomy_simple.json")
        logger.info("\nNext steps:")
        logger.info("1. Update config.yaml to enable Stage 2")
        logger.info("2. Run main.py to test species classification")
    else:
        logger.error("Failed to download taxonomy")
