import os
import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define input and output directories
input_dir = "data/structured_biographies"
output_dir = "data/occupations_for_classification"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

swedish_occupations = []
english_occupations = []

# Process each JSON file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(input_dir, filename)
        logging.info(f"Processing file: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract individual's occupation (now in person.occupation)
            if 'person' in data and 'occupation' in data['person']:
                occupation_info = data['person'].get('occupation', {})
                swedish_occ = occupation_info.get('occupation')
                english_occ = occupation_info.get('occupation_english')
                
                if swedish_occ:
                    swedish_occupations.append(swedish_occ)
                else:
                    logging.warning(f"Missing Swedish occupation in file {filename}")
                if english_occ:
                    english_occupations.append(english_occ)
                else:
                    logging.warning(f"Missing English occupation in file {filename}")
            
            # Extract parents' occupations (now in person.parents)
            if 'person' in data and 'parents' in data['person']:
                parents = data['person'].get('parents') or []
                if not parents:
                    logging.warning(f"No parent information in file {filename}")
                for parent in parents:
                    parent_occupation_info = parent.get('occupation') or {}
                    if parent_occupation_info:
                        parent_swedish_occ = parent_occupation_info.get('occupation')
                        parent_english_occ = parent_occupation_info.get('occupation_english')
                        
                        if parent_swedish_occ:
                            swedish_occupations.append(parent_swedish_occ)
                        else:
                            logging.warning(f"Parent missing Swedish occupation in file {filename}")
                        if parent_english_occ:
                            english_occupations.append(parent_english_occ)
                        else:
                            logging.warning(f"Parent missing English occupation in file {filename}")
                    else:
                        logging.warning(f"Parent occupation missing in file {filename}")
            
            # Extract spouse's occupation (in family.spouse.occupation)
            if 'family' in data and data['family'] and 'spouse' in data['family'] and data['family']['spouse']:
                spouse_occupation_info = data['family']['spouse'].get('occupation') or {}
                if spouse_occupation_info:
                    spouse_swedish_occ = spouse_occupation_info.get('occupation')
                    spouse_english_occ = spouse_occupation_info.get('occupation_english')
                    
                    if spouse_swedish_occ:
                        swedish_occupations.append(spouse_swedish_occ)
                    if spouse_english_occ:
                        english_occupations.append(spouse_english_occ)
            
            # Extract spouse's parents' occupations (in family.spouse.parents)
            if ('family' in data and data['family'] and 'spouse' in data['family'] and 
                data['family']['spouse'] and 'parents' in data['family']['spouse']):
                spouse_parents = data['family']['spouse'].get('parents') or []
                for spouse_parent in spouse_parents:
                    spouse_parent_occupation_info = spouse_parent.get('occupation') or {}
                    if spouse_parent_occupation_info:
                        spouse_parent_swedish_occ = spouse_parent_occupation_info.get('occupation')
                        spouse_parent_english_occ = spouse_parent_occupation_info.get('occupation_english')
                        
                        if spouse_parent_swedish_occ:
                            swedish_occupations.append(spouse_parent_swedish_occ)
                        if spouse_parent_english_occ:
                            english_occupations.append(spouse_parent_english_occ)
        
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

# Remove duplicates by converting lists to sets
swedish_occupations = list(set(swedish_occupations))
english_occupations = list(set(english_occupations))

# Create DataFrames
df_swedish = pd.DataFrame({'occupation': swedish_occupations})
df_english = pd.DataFrame({'occupation': english_occupations})

# Save to parquet files
output_path_swedish = os.path.join(output_dir, 'swedish_occupations.parquet')
output_path_english = os.path.join(output_dir, 'english_occupations.parquet')

try:
    df_swedish.to_parquet(output_path_swedish)
    logging.info(f"Swedish occupations saved to {output_path_swedish}")
except Exception as e:
    logging.error(f"Error saving Swedish occupations: {e}")

try:
    df_english.to_parquet(output_path_english)
    logging.info(f"English occupations saved to {output_path_english}")
except Exception as e:
    logging.error(f"Error saving English occupations: {e}")

logging.info(f"Total unique Swedish occupations: {len(swedish_occupations)}")
logging.info(f"Total unique English occupations: {len(english_occupations)}")