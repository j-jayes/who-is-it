import os
import json
import pandas as pd
from glob import glob
import logging

# Set up logging
logging.basicConfig(
    filename='data/occupations_for_classification/occupations_classified/data_processing.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Ensure the output directory exists
output_dir = "data/enriched_biographies_with_hisco_codes"    
os.makedirs(output_dir, exist_ok=True)

# Load classification data
try:
    english_classification = pd.read_parquet(
        "data/occupations_for_classification/occupations_classified/english_occupations_classified_joined.parquet"
    )
    swedish_classification = pd.read_parquet(
        "data/occupations_for_classification/occupations_classified/swedish_occupations_classified_joined.parquet"
    )
    logging.info("Classification data loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load classification data: {e}")
    raise e

# Convert classification data to dictionaries for faster lookup
english_classification_dict = english_classification.set_index('occupation').to_dict('index')
swedish_classification_dict = swedish_classification.set_index('occupation').to_dict('index')

# Function to enrich occupation data separately for English and Swedish
def enrich_occupation(occupation_data, english_dict, swedish_dict):
    """
    This module provides functionality to enrich occupation data with HISCO codes and descriptions
    based on Swedish and English occupation titles.

    Functions:
        enrich_occupation(occupation_data, english_dict, swedish_dict):
            Enriches the given occupation data with HISCO codes and descriptions if the occupation
            titles are found in the provided classification dictionaries and have a probability
            greater than 0.75.

            Parameters:
                occupation_data (dict): A dictionary containing occupation data with keys 'occupation'
                                        for Swedish title and 'occupation_english' for English title.
                english_dict (dict): A dictionary containing English occupation classifications with
                                     HISCO codes, descriptions, and probabilities.
                swedish_dict (dict): A dictionary containing Swedish occupation classifications with
                                     HISCO codes, descriptions, and probabilities.

            Returns:
                dict: The enriched occupation data with additional HISCO codes and descriptions if
                      applicable.
    """
    if occupation_data:
        # Enrich Swedish occupation
        swedish_title = occupation_data.get('occupation', '')
        if swedish_title:
            if swedish_title in swedish_dict:
                classification = swedish_dict[swedish_title]
                if classification['prob_1'] > 0.75:
                    occupation_data['hisco_code_swedish'] = classification['hisco_1']
                    occupation_data['hisco_description_swedish'] = classification['desc_1']
                    logging.info(f"Enriched Swedish occupation '{swedish_title}' with HISCO code {classification['hisco_1']}.")
                else:
                    logging.info(f"Swedish occupation '{swedish_title}' has prob_1 <= 0.75; skipping HISCO enrichment.")
            else:
                logging.warning(f"Swedish occupation '{swedish_title}' not found in classification data.")

        # Enrich English occupation
        english_title = occupation_data.get('occupation_english', '')
        if english_title:
            if english_title in english_dict:
                classification = english_dict[english_title]
                if classification['prob_1'] > 0.75:
                    occupation_data['hisco_code_english'] = classification['hisco_1']
                    occupation_data['hisco_description_english'] = classification['desc_1']
                    logging.info(f"Enriched English occupation '{english_title}' with HISCO code {classification['hisco_1']}.")
                else:
                    logging.info(f"English occupation '{english_title}' has prob_1 <= 0.75; skipping HISCO enrichment.")
            else:
                logging.warning(f"English occupation '{english_title}' not found in classification data.")
    return occupation_data

# Process each JSON file
json_files = glob("data/structured_biographies/*.json")

for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Processing file: {file_path}")

        # Enrich main person's occupation
        if 'person' in data and 'occupation' in data['person']:
            occupation_data = data['person'].get('occupation', {})
            data['person']['occupation'] = enrich_occupation(
                occupation_data,
                english_classification_dict,
                swedish_classification_dict
            )

        # Enrich parents' occupations
        if 'person' in data and 'parents' in data['person']:
            parents = data['person'].get('parents', [])
            if parents and isinstance(parents, list):
                for parent in parents:
                    parent_occupation_data = parent.get('occupation', {})
                    if parent_occupation_data:
                        parent['occupation'] = enrich_occupation(
                            parent_occupation_data,
                            english_classification_dict,
                            swedish_classification_dict
                        )
            else:
                logging.info(f"No parent occupation data found in file: {file_path}")

        # Enrich spouse's occupation (if present)
        if 'family' in data and 'spouse' in data['family'] and data['family']['spouse']:
            spouse_occupation_data = data['family']['spouse'].get('occupation', {})
            if spouse_occupation_data:
                data['family']['spouse']['occupation'] = enrich_occupation(
                    spouse_occupation_data,
                    english_classification_dict,
                    swedish_classification_dict
                )

        # Enrich spouse's parents' occupations (if present)
        if ('family' in data and 'spouse' in data['family'] and data['family']['spouse'] and
                'parents' in data['family']['spouse']):
            spouse_parents = data['family']['spouse'].get('parents', [])
            if spouse_parents and isinstance(spouse_parents, list):
                for spouse_parent in spouse_parents:
                    spouse_parent_occupation_data = spouse_parent.get('occupation', {})
                    if spouse_parent_occupation_data:
                        spouse_parent['occupation'] = enrich_occupation(
                            spouse_parent_occupation_data,
                            english_classification_dict,
                            swedish_classification_dict
                        )

        # Write back to a new JSON file in the output directory
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Enriched data written to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to process file {file_path}: {e}")