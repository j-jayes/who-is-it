import os
import json
import pandas as pd # Used for easier data handling, though not strictly necessary
from glob import glob
import logging
from collections import Counter # Efficiently counts frequencies
from typing import Dict, List, Any, Optional, Tuple

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
# Define directories & Paths (User should verify these)
INPUT_DIR: str = "data/enriched_biographies" 
OUTPUT_DIR: str = "data/analysis"
INSTITUTION_LIST_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'unique_education_institutions.json')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HISCO code ranges (same as previous script)
ENGINEER_HISCO_RANGES: List[Tuple[int, int]] = [(1000, 3999)] 
DIRECTOR_HISCO_RANGES: List[Tuple[int, int]] = [(21000, 21900)] 
RELEVANT_HISCO_RANGES: List[Tuple[int, int]] = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES

# --- Helper Functions (Subset needed from previous script) ---

def safe_float_to_int(value: Any) -> Optional[int]:
    """Safely convert a value (potentially float string) to integer."""
    if value is None: return None
    try: return int(float(value))
    except (ValueError, TypeError): return None

def check_hisco_range(code_str: Optional[str], ranges: List[Tuple[int, int]]) -> bool:
    """Check if a HISCO code string falls within specified numerical ranges."""
    code_value = safe_float_to_int(code_str)
    if code_value is None: return False
    for min_val, max_val in ranges:
        if min_val <= code_value <= max_val: return True
    return False

def is_relevant_person(person_data: Dict[str, Any]) -> bool:
    """Check if the person's occupation falls within relevant HISCO ranges."""
    occupation = person_data.get("occupation", {})
    if not occupation: return False
    hisco_code = occupation.get("hisco_code_swedish") or occupation.get("hisco_code_english")
    return check_hisco_range(hisco_code, RELEVANT_HISCO_RANGES)

# --- Main Function ---

def extract_unique_institutions() -> None:
    """
    Extracts unique educational institutions and their frequencies 
    for relevant persons (engineers/directors) from JSON biography files.
    """
    institution_counts = Counter()
    files_processed: int = 0
    persons_processed: int = 0

    # Check input directory
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return

    all_files: List[str] = glob(os.path.join(INPUT_DIR, "*.json"))
    logging.info(f"Found {len(all_files)} potential JSON files in {INPUT_DIR}")

    # Process each JSON file
    for filename in all_files:
        files_processed += 1
        if files_processed % 500 == 0: 
            logging.info(f"Processed {files_processed}/{len(all_files)} files...")

        try:
            with open(filename, 'r', encoding='utf-8') as f: 
                data = json.load(f)
            
            person = data.get('person', {})
            # Skip if no person data or person is not relevant
            if not person or not is_relevant_person(person): 
                continue 

            persons_processed += 1
            education_entries: List[Dict] = data.get('education', []) or []

            # Iterate through education entries for this person
            for edu_entry in education_entries:
                institution_name = edu_entry.get("institution")
                # Check if institution name is valid (not None, not empty string)
                if institution_name and isinstance(institution_name, str) and institution_name.strip() and institution_name.strip().lower() != 'none':
                    # Normalize slightly (e.g., strip whitespace) before counting
                    normalized_name = institution_name.strip() 
                    institution_counts[normalized_name] += 1
                    
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping file {os.path.basename(filename)} due to JSON decode error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing file {os.path.basename(filename)}: {e}", exc_info=False)

    logging.info(f"Finished processing files. Analyzed {persons_processed} relevant persons.")

    # --- Prepare and Save Output ---
    if not institution_counts:
        logging.warning("No valid educational institutions found.")
        return

    # Convert Counter to list of dicts, sorted by frequency
    output_list = [
        {"institution_name": name, "count": count} 
        for name, count in institution_counts.items()
    ]
    output_list.sort(key=lambda x: x['count'], reverse=True)

    logging.info(f"Found {len(output_list)} unique institution names.")

    # Save the list to a JSON file
    try:
        with open(INSTITUTION_LIST_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved unique institution list to: {INSTITUTION_LIST_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save output JSON file: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    extract_unique_institutions()
    logging.info("Script finished.")