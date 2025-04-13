import os
import json
import pandas as pd
# No geopandas needed for this script
from glob import glob
import logging
from collections import defaultdict # Useful for building the lookup
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np # Required for NaN handling

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
# Define directories & Paths (User should verify these)
OUTPUT_DIR: str = "data/analysis"
# **** INPUT: The CSV created by the previous script ****
INPUT_CSV_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_final_mapped_std_edu.csv') 
# **** OUTPUT: New CSV with network variables ****
NETWORK_CSV_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_with_networks.csv') 

# Network calculation parameters
# **** CHANGED COHORT WINDOW FROM 2 to 4 ****
COHORT_YEAR_WINDOW: int = 4 # Years before/after to consider someone part of the cohort (+/-)

# --- Helper Functions ---
def parse_json_string(json_string: Optional[str], default_value: Any = []) -> Any:
    """Safely parse a JSON string, returning a default value on error."""
    if pd.isna(json_string) or not isinstance(json_string, str):
        return default_value
    try:
        # Added check for empty string after potential NA check
        if not json_string.strip(): 
            return default_value
        return json.loads(json_string)
    except json.JSONDecodeError:
        # logging.warning(f"Could not parse JSON string: {json_string[:100]}...")
        return default_value

# --- Main Network Calculation Logic ---

def calculate_education_networks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates educational network variables based on shared institution and time.

    Args:
        df: DataFrame containing person data, including parsed education history
            and 'birth_parish_is_western_line'.

    Returns:
        DataFrame with added network variable columns.
    """
    logging.info("Building institution-student lookup...")
    # Structure: {institution_standardized: [(person_id, year, born_wl_bool), ...]}
    institution_lookup = defaultdict(list)
    
    # Check if required columns exist
    required_cols = ['education_standardized_parsed', 'person_id', 'birth_parish_is_western_line']
    if not all(col in df.columns for col in required_cols):
         logging.error(f"Input DataFrame missing required columns for network calculation. Need: {required_cols}. Found: {list(df.columns)}")
         # Return df with empty columns
         df['edu_network_size'] = 0
         df['edu_network_wl_birth_prop'] = np.nan
         return df

    # Populate the lookup table
    for idx, row in df.iterrows():
        person_id = row['person_id']
        # Use boolean value directly (True/False/None)
        born_wl = row['birth_parish_is_western_line'] 
        
        education_list = row.get('education_standardized_parsed', [])
        if not isinstance(education_list, list): continue # Skip if not a list

        for edu_entry in education_list:
             inst_std = edu_entry.get("institution_standardized")
             # Use the original 'year' field for cohort calculation
             year_str = edu_entry.get("year") 
             year = pd.to_numeric(year_str, errors='coerce')
             
             # Only consider entries with a valid standardized institution and year
             if pd.notna(inst_std) and pd.notna(year) and isinstance(inst_std, str):
                 # Ensure year is integer for cleaner handling
                 institution_lookup[inst_std].append((person_id, int(year), born_wl))

    logging.info(f"Institution lookup built with {len(institution_lookup)} institutions.")

    # --- Calculate network variables for each person ---
    network_sizes = []
    network_wl_props = []

    logging.info("Calculating network variables for each person...")
    for idx, row in df.iterrows():
        person_id = row['person_id']
        max_network_size = 0
        # Store proportions for averaging later (list of valid proportions found)
        valid_wl_proportions = [] 
        
        education_list = row.get('education_standardized_parsed', [])
        if not isinstance(education_list, list): 
             network_sizes.append(0)
             network_wl_props.append(np.nan)
             continue # Skip if education data is invalid

        # --- Logic to find highest/primary degree (example - can be refined) ---
        # This example finds the latest graduation year among entries with institution
        latest_grad_year = -1
        primary_edu_entries = []
        
        temp_primary_edu = []
        for edu_entry in education_list:
            inst_std = edu_entry.get("institution_standardized")
            year_str = edu_entry.get("year")
            year = pd.to_numeric(year_str, errors='coerce')
            if pd.notna(inst_std) and pd.notna(year) and isinstance(inst_std, str):
                year = int(year)
                temp_primary_edu.append({'inst': inst_std, 'year': year})
                if year > latest_grad_year:
                    latest_grad_year = year
        
        # Keep only entries from the latest year(s) found
        if latest_grad_year > 0:
             primary_edu_entries = [edu for edu in temp_primary_edu if edu['year'] == latest_grad_year]
        # If no valid entries found, primary_edu_entries remains empty
        # --- End Example Logic ---

        # Calculate network based on the identified primary education entries
        # If multiple entries have the same latest year, stats are calculated for all of them
        # and then max size / avg proportion is taken.
        if not primary_edu_entries: # If no valid education found for person
             network_sizes.append(0)
             network_wl_props.append(np.nan)
             continue

        person_cohort_sizes = []
        person_cohort_wl_props = []

        for primary_edu in primary_edu_entries:
             inst_std = primary_edu['inst']
             year = primary_edu['year']
             
             cohort = []
             # Find peers in the lookup table
             if inst_std in institution_lookup:
                 min_year = year - COHORT_YEAR_WINDOW # Use updated window
                 max_year = year + COHORT_YEAR_WINDOW # Use updated window
                 
                 for peer_id, peer_year, peer_born_wl in institution_lookup[inst_std]:
                     # Check year window and exclude the person themselves
                     if min_year <= peer_year <= max_year and peer_id != person_id:
                         cohort.append({'id': peer_id, 'born_wl': peer_born_wl})
             
             current_network_size = len(cohort)
             person_cohort_sizes.append(current_network_size)

             # Calculate WL proportion for this specific cohort
             if current_network_size > 0:
                 wl_peers = sum(1 for peer in cohort if peer['born_wl'] is True) 
                 wl_proportion = wl_peers / current_network_size
                 person_cohort_wl_props.append(wl_proportion)

        # Assign max size and average proportion from the primary education entries
        network_sizes.append(max(person_cohort_sizes) if person_cohort_sizes else 0)
        network_wl_props.append(np.mean(person_cohort_wl_props) if person_cohort_wl_props else np.nan)

    # Add calculated lists as new columns
    df['edu_network_size'] = network_sizes
    df['edu_network_wl_birth_prop'] = network_wl_props
    
    logging.info("Finished calculating network variables.")
    return df


# --- Main Execution ---
def main() -> int:
    """Main execution function."""
    logging.info(f"Starting network variable calculation...")
    
    # --- 1. Load Previous Results ---
    if not os.path.exists(INPUT_CSV_PATH):
        logging.error(f"Input CSV file not found: {INPUT_CSV_PATH}. Run the previous script first.")
        return 1
    try:
        df = pd.read_csv(INPUT_CSV_PATH, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'])
        logging.info(f"Loaded data ({len(df)} rows) from: {INPUT_CSV_PATH}")
    except Exception as e:
        logging.error(f"Failed to load input CSV: {e}")
        return 1

    # --- 2. Parse JSON Column ---
    logging.info("Parsing standardized education JSON strings...")
    if 'education_standardized' in df.columns:
         # Use the helper function for safe parsing
         df['education_standardized_parsed'] = df['education_standardized'].apply(parse_json_string) 
    else:
         logging.error("Column 'education_standardized' not found. Cannot calculate networks.")
         return 1

    # --- 3. Calculate Network Variables ---
    df_with_networks = calculate_education_networks(df)
    
    # Remove temporary parsed column
    df_with_networks = df_with_networks.drop(columns=['education_standardized_parsed'], errors='ignore')

    # --- 4. Save Results ---
    try:
        # Save with utf-8-sig encoding
        df_with_networks.to_csv(NETWORK_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig') 
        logging.info(f"Saved final data with network variables ({len(df_with_networks)} rows) to: {NETWORK_CSV_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save final CSV with network variables: {e}")
        return 1
        
    logging.info("Network calculation script finished successfully.")
    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)