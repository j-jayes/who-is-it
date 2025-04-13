import os
import json
import pandas as pd
from glob import glob
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories
# Make sure these paths are correct for your environment
INPUT_DIR = "data/enriched_biographies" 
OUTPUT_DIR = "data/analysis"
PARISH_SHAPEFILE_PATH = "data/parishes/parish_map_1920.shp" # Path mentioned but not used in this script

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
# HISCO code ranges (adjust as needed based on your data's format and desired scope)
# User specified 1000-3999 for engineers. Note: Standard HISCO major groups are 02/03 for Eng/Tech.
# Assuming 5-digit codes derived from floats (e.g., 21230.0)
ENGINEER_HISCO_RANGES = [(1000, 3999)] # Captures a broad range, adjust if needed (e.g., [(2000, 3999)])
DIRECTOR_HISCO_RANGES = [(21000, 21900)] # Example: Managers (may need broader definition, e.g., 1xxxx for higher managers)
RELEVANT_HISCO_RANGES = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES

MAX_EARLY_CAREER_ENTRIES = 3 # How many first jobs to extract for location analysis

# Educational institution classification keywords (keep as is or refine)
TECHNICAL_KEYWORDS = [
    'tekniska', 'chalmers', 'kth', 'tekn', 'ingenjör', 'teknisk', 
    'teknolog', 'polytekn', 'engineering', 'technical'
]
BUSINESS_KEYWORDS = [
    'handels', 'ekonomi', 'handelshögskola', 'business', 'commerce', 
    'ekonomisk', 'handelsinstitut', 'handelsgymnasium', 'ekonom'
]
# --- Helper Functions ---

def safe_float_to_int(value: Any) -> Optional[int]:
    """Safely convert a value (potentially float string) to int."""
    if value is None:
        return None
    try:
        # Handle potential float strings like "21230.0"
        return int(float(value))
    except (ValueError, TypeError):
        return None

def check_hisco_range(code_str: Optional[str], ranges: List[Tuple[int, int]]) -> bool:
    """Check if a HISCO code string falls within specified ranges."""
    code_value = safe_float_to_int(code_str)
    if code_value is None:
        return False
    
    for min_val, max_val in ranges:
        if min_val <= code_value <= max_val:
            return True
    return False

def is_relevant_person(person_data: Dict[str, Any]) -> bool:
    """Check if the person is an engineer or director based on HISCO code."""
    occupation = person_data.get("occupation", {})
    if not occupation: return False # Added check if occupation exists
        
    # Check Swedish HISCO code
    swedish_code = occupation.get("hisco_code_swedish")
    if check_hisco_range(swedish_code, RELEVANT_HISCO_RANGES):
        return True
        
    # Check English HISCO code if Swedish code didn't match
    english_code = occupation.get("hisco_code_english")
    if check_hisco_range(english_code, RELEVANT_HISCO_RANGES):
        return True
        
    # Fallback: Check occupation title for keywords if no valid HISCO
    # This is less reliable, use with caution or disable if needed
    occ_title = str(occupation.get('occupation', '')).lower()
    if any(keyword in occ_title for keyword in ['ingenjör', 'engineer', 'direktör', 'director', 'manager', 'chef']):
         # Check if it's explicitly NOT one of the target groups if possible
         # Add more nuanced checks if needed
         # logging.warning(f"Person {person_data.get('first_name')} {person_data.get('last_name')} identified by title keyword: {occ_title}")
         # Consider returning False here if HISCO should be the only criteria
         pass # Currently allows fallback, comment out `pass` and uncomment `return False` below to disable
         # return False 
            
    return False

def extract_birth_decade(birth_date: Optional[str]) -> Optional[int]:
    """Extract the decade from a birth date string."""
    if not birth_date or not isinstance(birth_date, str):
        return None
    
    year = None
    # Try YYYY-MM-DD or YYYYMMDD
    year_match = re.match(r'^(\d{4})', birth_date)
    if year_match:
        year = int(year_match.group(1))
    else:
        # Try DD-MM-YYYY
        parts = birth_date.split('-')
        if len(parts) == 3 and len(parts[2]) >= 4: # Check if year part has at least 4 digits
             year_match_end = re.search(r'(\d{4})$', parts[2])
             if year_match_end:
                 year = int(year_match_end.group(1))

    if year:
        # Basic sanity check for year range
        if 1700 < year < 2000:
             return (year // 10) * 10
        else:
             # logging.warning(f"Extracted year {year} seems out of range for birth date {birth_date}")
             return None # Or handle unreasonable years differently
    else:
         # logging.warning(f"Could not extract year from birth date: {birth_date}")
         return None


def classify_education(education_entries: Optional[List[Dict[str, Any]]]) -> Dict[str, bool]:
    """Classify education entries by institution type."""
    if not education_entries:
        return {"technical": False, "business": False, "other_higher": False}
    
    result = {"technical": False, "business": False, "other_higher": False}
    has_higher_edu = False
    
    for entry in education_entries:
        institution = str(entry.get("institution", "")).lower()
        degree = str(entry.get("degree", "")).lower()
        full_text = institution + " " + degree # Combine for keyword search
        
        # Check if it's potentially higher education
        is_higher = entry.get("degree_level") in ["Master's", "PhD", "Bachelor's", "Licentiate"] or \
                    any(keyword in full_text for keyword in ['högskola', 'universitet', 'university', 'college'])

        if is_higher:
            has_higher_edu = True # Mark that at least one higher education entry exists
        
        # Check if any technical keywords appear
        if any(keyword in full_text for keyword in TECHNICAL_KEYWORDS):
            result["technical"] = True
        
        # Check if any business keywords appear
        elif any(keyword in full_text for keyword in BUSINESS_KEYWORDS):
            result["business"] = True
        
        # If it is higher education but not classified yet, mark as other_higher
        elif is_higher:
             result["other_higher"] = True

    # Refinement: If only lower level degrees found, ensure 'other_higher' is False
    if not has_higher_edu:
        result["other_higher"] = False
            
    return result

def get_location_details(location_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely extract location details."""
    if not location_data or not isinstance(location_data, dict):
        return {"name": None, "lat": None, "lon": None, "formatted": None}
    return {
        "name": location_data.get("name"),
        "lat": location_data.get("latitude"),
        "lon": location_data.get("longitude"),
        "formatted": location_data.get("formatted_address")
    }

def get_overseas_experience(career_entries: Optional[List[Dict[str, Any]]]) -> Tuple[bool, bool, List[str]]:
    """Check for overseas and US experience from career entries."""
    has_overseas = False
    has_us = False
    countries = set()
    
    if not career_entries:
        return has_overseas, has_us, list(countries)
        
    for entry in career_entries:
        country_code = entry.get("country_code")
        if country_code and isinstance(country_code, str) and country_code.upper() != "SWE":
            has_overseas = True
            countries.add(country_code.upper())
            if country_code.upper() == "USA":
                has_us = True
                
    return has_overseas, has_us, sorted(list(countries))

# --- Main Extraction Function ---

def extract_persons_data():
    """Extract key information for relevant persons (engineers/directors)."""
    persons_data = []
    files_processed = 0
    persons_count = 0
    relevant_files_count = 0

    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return []

    all_files = glob(os.path.join(INPUT_DIR, "*.json"))
    logging.info(f"Found {len(all_files)} potential JSON files in {INPUT_DIR}")

    # Process each JSON file in the input directory
    for filename in all_files:
        files_processed += 1
        
        if files_processed % 500 == 0:
             logging.info(f"Processed {files_processed}/{len(all_files)} files...")

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            person = data.get('person', {})
            if not person:
                 # logging.warning(f"Skipping file {os.path.basename(filename)}: Missing 'person' data.")
                 continue # Skip if no person data

            # Check if the person is relevant (engineer or director based on HISCO)
            if is_relevant_person(person):
                persons_count += 1
                relevant_files_count += 1
                
                # --- Extract Person Info ---
                person_name = f"{person.get('first_name', '')} {person.get('last_name', '')}".strip()
                birth_date = person.get('birth_date')
                birth_decade = extract_birth_decade(birth_date)
                birth_loc = get_location_details(person.get('birth_place'))
                occupation = person.get('occupation', {}) or {}

                # --- Extract Father Info ---
                father = {}
                parents = person.get('parents', [])
                if parents and isinstance(parents, list):
                    for p in parents:
                        if p and p.get('gender') == 'Male': # Assuming father is male parent listed
                            father = {
                                'name': f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                                'occupation': (p.get('occupation') or {}).get('occupation'),
                                'hisco_code': (p.get('occupation') or {}).get('hisco_code_swedish') or (p.get('occupation') or {}).get('hisco_code_english') # Prioritize Swedish
                            }
                            break # Take the first male parent found

                # --- Extract Education Info ---
                education_entries = data.get('education', []) or []
                education_classification = classify_education(education_entries)
                education_details = []
                for edu in education_entries:
                    edu_loc = get_location_details(edu.get('location')) # Assuming location exists per entry
                    education_details.append({
                        "institution": edu.get("institution"),
                        "degree": edu.get("degree"),
                        "degree_level": edu.get("degree_level"),
                        "year": edu.get("year"),
                        "location_name": edu_loc["name"],
                        "location_lat": edu_loc["lat"],
                        "location_lon": edu_loc["lon"],
                    })

                # --- Extract Career & Overseas Info ---
                career_entries = data.get('career', []) or []
                has_overseas, has_us, overseas_countries = get_overseas_experience(career_entries)
                
                early_career = []
                # Sort career entries by start year to get the earliest ones
                sorted_career = sorted([c for c in career_entries if c.get('start_year')], key=lambda x: x['start_year'])
                
                for i, job in enumerate(sorted_career):
                    if i >= MAX_EARLY_CAREER_ENTRIES:
                        break
                    job_loc = get_location_details(job.get('location'))
                    early_career.append({
                        "position": job.get("position"),
                        "organization": job.get("organization"),
                        "start_year": job.get("start_year"),
                        "end_year": job.get("end_year"),
                        "location_name": job_loc["name"],
                        "location_lat": job_loc["lat"],
                        "location_lon": job_loc["lon"],
                        "country_code": job.get("country_code")
                    })
                    
                # --- Extract Board Info ---
                board_memberships = data.get('board_memberships', []) or []
                board_count = len(board_memberships)

                # --- Create Record ---
                person_record = {
                    'person_id': os.path.basename(filename).replace('.json', ''),
                    'person_name': person_name,
                    'birth_date': birth_date,
                    'birth_decade': birth_decade,
                    'birth_location_name': birth_loc["name"],
                    'birth_location_lat': birth_loc["lat"],
                    'birth_location_lon': birth_loc["lon"],
                    'person_occupation': occupation.get('occupation'),
                    'person_hisco_swe': occupation.get('hisco_code_swedish'),
                    'person_hisco_eng': occupation.get('hisco_code_english'),
                    
                    'father_name': father.get('name'),
                    'father_occupation': father.get('occupation'),
                    'father_hisco': father.get('hisco_code'),
                    
                    'edu_technical': education_classification['technical'],
                    'edu_business': education_classification['business'],
                    'edu_other_higher': education_classification['other_higher'],
                    'education_details': json.dumps(education_details), # Store complex list as JSON string
                    
                    'career_has_overseas': has_overseas,
                    'career_has_us': has_us,
                    'career_overseas_countries': ",".join(overseas_countries), # Store list as comma-separated string
                    'early_career_details': json.dumps(early_career), # Store complex list as JSON string
                    
                    'board_membership_count': board_count,
                    
                    # Placeholder columns for results of external parish mapping
                    'birth_parish_code': None,
                    'birth_parish_name': None,
                    'birth_parish_is_western_line': None,
                    'early_career_parish_code': None, # To be determined based on rules
                    'early_career_parish_name': None,
                    'early_career_parish_is_western_line': None,
                }
                
                persons_data.append(person_record)
            else:
                 # Optionally log skipped persons if needed for debugging
                 # logging.debug(f"Skipping file {os.path.basename(filename)}: Person {person.get('first_name')} {person.get('last_name')} not in relevant HISCO range.")
                 pass

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON file {os.path.basename(filename)}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing file {os.path.basename(filename)}: {e}", exc_info=True) # Added traceback info
    
    logging.info(f"Completed processing {files_processed} files.")
    logging.info(f"Found {persons_count} relevant persons (Engineers/Directors based on HISCO ranges {RELEVANT_HISCO_RANGES}).")
    return persons_data

# --- Analysis/Metadata Functions (Kept similar to original for continuity) ---

def analyze_education_by_decade(persons_data: List[Dict[str, Any]]):
    """Analyze education distribution by birth decade for relevant persons."""
    if not persons_data:
        logging.warning("No persons data to analyze by decade.")
        return pd.DataFrame()
        
    df = pd.DataFrame(persons_data)
    
    # Filter out persons with missing birth decade
    df_with_decade = df.dropna(subset=['birth_decade']).copy() # Use .copy() to avoid SettingWithCopyWarning
    # Ensure boolean columns are treated as numeric for aggregation
    bool_cols = ['edu_technical', 'edu_business', 'edu_other_higher']
    for col in bool_cols:
        df_with_decade[col] = df_with_decade[col].astype(int)

    if df_with_decade.empty:
        logging.warning("No persons data with valid birth decades found.")
        return pd.DataFrame()

    # Group by birth decade
    decade_stats = df_with_decade.groupby('birth_decade').agg(
        total_persons=('person_id', 'count'),
        sum_technical=('edu_technical', 'sum'),
        sum_business=('edu_business', 'sum'),
        sum_other_higher=('edu_other_higher', 'sum')
    ).reset_index()
    
    # Calculate percentages
    decade_stats['pct_technical'] = (decade_stats['sum_technical'] / decade_stats['total_persons'] * 100).round(1)
    decade_stats['pct_business'] = (decade_stats['sum_business'] / decade_stats['total_persons'] * 100).round(1)
    decade_stats['pct_other_higher'] = (decade_stats['sum_other_higher'] / decade_stats['total_persons'] * 100).round(1)
    
    # Clean up column names
    decade_stats = decade_stats.rename(columns={'birth_decade': 'decade'})
    
    return decade_stats

def extract_education_metadata(persons_data: List[Dict[str, Any]]):
    """Extract metadata about education institutions/degrees for manual review."""
    education_institutions = defaultdict(int)
    education_degrees = defaultdict(int)
    
    if not persons_data:
        return {'institutions': [], 'degrees': []}

    for person in persons_data:
        try:
            # Education details are stored as a JSON string, parse it back
            edu_details = json.loads(person.get('education_details', '[]'))
            if not isinstance(edu_details, list): continue # Skip if not a list
                
            for edu in edu_details:
                 inst = edu.get('institution')
                 deg = edu.get('degree')
                 if inst:
                     education_institutions[inst] += 1
                 if deg:
                     education_degrees[deg] += 1
        except json.JSONDecodeError:
            logging.warning(f"Could not parse education_details for person_id {person.get('person_id')}")
        except Exception as e:
            logging.error(f"Error processing education metadata for person_id {person.get('person_id')}: {e}")

    
    # Convert to sorted lists of dicts
    institutions_list = [{'institution': k, 'count': v} for k, v in education_institutions.items()]
    institutions_list = sorted(institutions_list, key=lambda x: x['count'], reverse=True)
    
    degrees_list = [{'degree': k, 'count': v} for k, v in education_degrees.items()]
    degrees_list = sorted(degrees_list, key=lambda x: x['count'], reverse=True)
    
    return {
        'institutions': institutions_list,
        'degrees': degrees_list
    }

# --- Main Execution ---

def main():
    """Main execution function."""
    logging.info(f"Starting data extraction for Engineers/Directors (HISCO ranges: {RELEVANT_HISCO_RANGES})")
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # Extract data from all relevant biographies
    persons_data = extract_persons_data()
    
    if not persons_data:
        logging.warning("No relevant persons found or extracted. Exiting.")
        return 1
        
    # Save the primary data output for analysis
    persons_df = pd.DataFrame(persons_data)
    output_csv_path = os.path.join(OUTPUT_DIR, 'persons_data_for_analysis.csv')
    persons_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logging.info(f"Saved detailed data for {len(persons_df)} persons to {output_csv_path}")
    logging.info("NOTE: This CSV contains coordinates (lat/lon) for birth/education/career locations.")
    logging.info(f"      You need to perform spatial analysis separately using these coordinates and the parish shapefile ({PARISH_SHAPEFILE_PATH}) to determine Western Line status.")
    
    # Perform analysis by decade (optional, but kept from original structure)
    decade_stats = analyze_education_by_decade(persons_data)
    if not decade_stats.empty:
         decade_output_path = os.path.join(OUTPUT_DIR, 'persons_education_by_decade.csv')
         decade_stats.to_csv(decade_output_path, index=False, encoding='utf-8')
         logging.info(f"Saved education analysis by decade to {decade_output_path}")
         logging.info("\nEducation by Birth Decade Summary:")
         print(decade_stats[['decade', 'total_persons', 'pct_technical', 'pct_business', 'pct_other_higher']])
    else:
         logging.info("Skipped saving decade analysis due to lack of data.")

    # Extract education metadata for classification refinement (optional)
    education_metadata = extract_education_metadata(persons_data)
    metadata_output_path = os.path.join(OUTPUT_DIR, 'education_metadata.json')
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(education_metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved education metadata (institutions/degrees) to {metadata_output_path}")
    
    logging.info("Data extraction process finished.")
    return 0

if __name__ == "__main__":
    main() # Changed exit() call to just main() for better practice if run as script