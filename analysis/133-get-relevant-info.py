import os
import json
import pandas as pd
import geopandas as gpd # Added for spatial operations
from glob import glob
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np # For NaN handling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories
INPUT_DIR = "data/enriched_biographies" 
OUTPUT_DIR = "data/analysis"
PARISH_SHAPEFILE_PATH = "data/parishes/parish_map_1920.shp" 

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
ENGINEER_HISCO_RANGES = [(1000, 3999)] 
DIRECTOR_HISCO_RANGES = [(21000, 21900)] 
RELEVANT_HISCO_RANGES = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES
MAX_EARLY_CAREER_ENTRIES = 3 # Still used for 'early_career_details' column, but mapping uses all career entries

# --- Helper Functions (mostly unchanged, check previous versions if needed) ---
# safe_float_to_int, check_hisco_range, is_relevant_person, 
# extract_birth_decade, classify_education, get_location_details, 
# get_overseas_experience
# (Include the definitions for these helper functions as in the previous script)
def safe_float_to_int(value: Any) -> Optional[int]:
    """Safely convert a value (potentially float string) to int."""
    if value is None: return None
    try: return int(float(value))
    except (ValueError, TypeError): return None

def check_hisco_range(code_str: Optional[str], ranges: List[Tuple[int, int]]) -> bool:
    """Check if a HISCO code string falls within specified ranges."""
    code_value = safe_float_to_int(code_str)
    if code_value is None: return False
    for min_val, max_val in ranges:
        if min_val <= code_value <= max_val: return True
    return False

def is_relevant_person(person_data: Dict[str, Any]) -> bool:
    """Check if the person is an engineer or director based on HISCO code."""
    occupation = person_data.get("occupation", {})
    if not occupation: return False
    swedish_code = occupation.get("hisco_code_swedish")
    if check_hisco_range(swedish_code, RELEVANT_HISCO_RANGES): return True
    english_code = occupation.get("hisco_code_english")
    if check_hisco_range(english_code, RELEVANT_HISCO_RANGES): return True
    return False

def extract_birth_decade(birth_date: Optional[str]) -> Optional[int]:
    """Extract the decade from a birth date string."""
    if not birth_date or not isinstance(birth_date, str): return None
    year = None
    year_match = re.match(r'^(\d{4})', birth_date)
    if year_match: year = int(year_match.group(1))
    else:
        parts = birth_date.split('-')
        if len(parts) == 3 and len(parts[2]) >= 4:
             year_match_end = re.search(r'(\d{4})$', parts[2])
             if year_match_end: year = int(year_match_end.group(1))
    if year and 1700 < year < 2000: return (year // 10) * 10
    return None

def classify_education(education_entries: Optional[List[Dict[str, Any]]]) -> Dict[str, bool]:
    """Classify education entries by institution type."""
    if not education_entries: return {"technical": False, "business": False, "other_higher": False}
    result = {"technical": False, "business": False, "other_higher": False}
    has_higher_edu = False
    # Simplified keywords from previous version - add back if needed
    TECHNICAL_KEYWORDS = ['tekniska', 'chalmers', 'kth', 'tekn', 'ingenjör', 'teknisk', 'teknolog', 'polytekn', 'engineering', 'technical']
    BUSINESS_KEYWORDS = ['handels', 'ekonomi', 'handelshögskola', 'business', 'commerce', 'ekonomisk', 'handelsinstitut', 'handelsgymnasium', 'ekonom']
    for entry in education_entries:
        institution = str(entry.get("institution", "")).lower()
        degree = str(entry.get("degree", "")).lower()
        full_text = institution + " " + degree
        is_higher = entry.get("degree_level") in ["Master's", "PhD", "Bachelor's", "Licentiate"] or \
                    any(keyword in full_text for keyword in ['högskola', 'universitet', 'university', 'college'])
        if is_higher: has_higher_edu = True
        if any(keyword in full_text for keyword in TECHNICAL_KEYWORDS): result["technical"] = True
        elif any(keyword in full_text for keyword in BUSINESS_KEYWORDS): result["business"] = True
        elif is_higher: result["other_higher"] = True
    if not has_higher_edu: result["other_higher"] = False
    return result

def get_location_details(location_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely extract location details."""
    if not location_data or not isinstance(location_data, dict):
        return {"name": None, "lat": None, "lon": None, "formatted": None}
    # Attempt to convert lat/lon to numeric early
    lat = pd.to_numeric(location_data.get("latitude"), errors='coerce')
    lon = pd.to_numeric(location_data.get("longitude"), errors='coerce')
    return {
        "name": location_data.get("name"),
        "lat": lat if pd.notna(lat) else None, # Store as None if not numeric
        "lon": lon if pd.notna(lon) else None,
        "formatted": location_data.get("formatted_address")
    }

def get_overseas_experience(career_entries: Optional[List[Dict[str, Any]]]) -> Tuple[bool, bool, List[str]]:
    """Check for overseas and US experience from career entries."""
    has_overseas, has_us, countries = False, False, set()
    if not career_entries: return has_overseas, has_us, list(countries)
    for entry in career_entries:
        country_code = entry.get("country_code")
        if country_code and isinstance(country_code, str) and country_code.upper() != "SWE":
            has_overseas = True
            countries.add(country_code.upper())
            if country_code.upper() == "USA": has_us = True
    return has_overseas, has_us, sorted(list(countries))

# --- Data Extraction Function (extracts raw data + coordinates) ---
def extract_persons_data() -> List[Dict]:
    """Extract key information into a list of dictionaries, including coordinates."""
    persons_data_list = []
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return []

    all_files = glob(os.path.join(INPUT_DIR, "*.json"))
    logging.info(f"Found {len(all_files)} potential JSON files.")
    files_processed = 0
    
    for filename in all_files:
        files_processed += 1
        if files_processed % 500 == 0: logging.info(f"Processed {files_processed}/{len(all_files)} files...")

        try:
            with open(filename, 'r', encoding='utf-8') as f: data = json.load(f)
            person = data.get('person', {})
            if not person or not is_relevant_person(person): continue 

            # Extract details including locations
            birth_loc = get_location_details(person.get('birth_place'))
            current_loc = get_location_details(data.get('current_location')) # Extract current location
            occupation = person.get('occupation', {}) or {}
            
            # Father info
            father = {}
            parents = person.get('parents', [])
            if parents:
                 for p in parents:
                     if p and p.get('gender') == 'Male':
                         father = {'name': f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                                   'occupation': (p.get('occupation') or {}).get('occupation'),
                                   'hisco_code': (p.get('occupation') or {}).get('hisco_code_swedish') or (p.get('occupation') or {}).get('hisco_code_english') }
                         break
                         
            # Education info (extracting raw for later processing if needed)
            education_entries = data.get('education', []) or []
            education_classification = classify_education(education_entries)
            
            # Career info (extracting raw for later processing)
            career_entries = data.get('career', []) or []
            has_overseas, has_us, overseas_countries = get_overseas_experience(career_entries)
            board_count = len(data.get('board_memberships', []) or [])

            person_record = {
                'person_id': os.path.basename(filename).replace('.json', ''),
                'person_name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                'birth_date': person.get('birth_date'),
                'birth_decade': extract_birth_decade(person.get('birth_date')),
                'birth_location_name': birth_loc["name"],
                'birth_location_lat': birth_loc["lat"], 
                'birth_location_lon': birth_loc["lon"], 
                
                # Add current location coordinates
                'current_location_name': current_loc["name"],
                'current_location_lat': current_loc["lat"],
                'current_location_lon': current_loc["lon"],
                
                'person_occupation': occupation.get('occupation'),
                'person_hisco_swe': occupation.get('hisco_code_swedish'),
                'person_hisco_eng': occupation.get('hisco_code_english'),
                'father_name': father.get('name'),
                'father_occupation': father.get('occupation'),
                'father_hisco': father.get('hisco_code'),
                'edu_technical': education_classification['technical'],
                'edu_business': education_classification['business'],
                'edu_other_higher': education_classification['other_higher'],
                'career_has_overseas': has_overseas,
                'career_has_us': has_us,
                'career_overseas_countries': ",".join(overseas_countries), 
                'board_membership_count': board_count,
                
                # Store raw education and career for later processing
                '_education_raw': education_entries, 
                '_career_raw': career_entries,
             }
            persons_data_list.append(person_record)
        except Exception as e: logging.error(f"Error processing {os.path.basename(filename)}: {e}", exc_info=False)

    logging.info(f"Completed initial data extraction. Found {len(persons_data_list)} relevant persons.")
    return persons_data_list

# --- Function to Map Unique Locations ---
def map_unique_locations(locations_df: pd.DataFrame, parishes_gdf: gpd.GeoDataFrame) -> Dict[Tuple[float, float], Dict]:
    """
    Maps unique lat/lon pairs to parish information.

    Args:
        locations_df: DataFrame with 'lat', 'lon' columns for unique locations.
        parishes_gdf: GeoDataFrame of parishes with CRS and info columns.

    Returns:
        Dictionary mapping (lat, lon) tuples to {'parish_code': ..., 'parish_name': ..., 'western_line': ...}
        Returns empty dict if mapping fails.
    """
    if locations_df.empty: return {}
    
    logging.info(f"Mapping {len(locations_df)} unique locations to parishes...")
    location_map = {}
    
    try:
        # Create GeoDataFrame for unique locations
        locations_gdf = gpd.GeoDataFrame(
            locations_df,
            geometry=gpd.points_from_xy(locations_df['lon'], locations_df['lat']),
            crs="EPSG:4326" # Assume WGS84 input
        )
        
        # Align CRS
        locations_gdf = locations_gdf.to_crs(parishes_gdf.crs)
        
        # Spatial Join
        # Ensure correct column names from YOUR shapefile are used here
        parishes_simplified = parishes_gdf[['geometry', 'prsh_cd', 'prsh_nm', 'wstrn_l']].copy() 
        joined_gdf = gpd.sjoin(locations_gdf, parishes_simplified, how='left', predicate='within')

        # Create lookup dictionary
        for idx, row in joined_gdf.iterrows():
            # Use original index from locations_df to get lat/lon
            original_idx = row.name # sjoin preserves original index
            lat, lon = locations_df.loc[original_idx, 'lat'], locations_df.loc[original_idx, 'lon']
            # Store parish info, handling potential NaN from left join
            location_map[(lat, lon)] = {
                'parish_code': row.get('prsh_cd'), # Use correct shapefile col name
                'parish_name': row.get('prsh_nm'), # Use correct shapefile col name
                'western_line': int(row['wstrn_l']) if pd.notna(row.get('wstrn_l')) else -1 # Use correct shapefile col name, handle NaN
            }
            
        logging.info(f"Finished mapping unique locations. Created lookup for {len(location_map)} coordinates.")
        
    except Exception as e:
        logging.error(f"Error during unique location mapping: {e}")
        return {} # Return empty dict on error

    return location_map


# --- Main Execution ---
def main():
    """Main execution function."""
    logging.info(f"Starting data extraction and mapping process...")
    
    # --- 1. Extract Raw Data ---
    persons_data_list = extract_persons_data()
    if not persons_data_list:
        logging.warning("No relevant persons found or extracted. Exiting.")
        return 1
        
    # --- 2. Load Parish Shapefile ---
    if not os.path.exists(PARISH_SHAPEFILE_PATH):
        logging.error(f"Parish shapefile not found at {PARISH_SHAPEFILE_PATH}. Cannot perform spatial mapping.")
        # Optionally save the raw extracted data without mapping
        # pd.DataFrame(persons_data_list).to_csv(os.path.join(OUTPUT_DIR, 'persons_data_raw_no_mapping.csv'), index=False)
        return 1
        
    try:
        parishes_gdf = gpd.read_file(PARISH_SHAPEFILE_PATH)
        # Ensure CRS is set (e.g., EPSG:3006) - handle missing CRS if necessary
        if not parishes_gdf.crs:
             logging.warning("Parish shapefile CRS missing. Assuming EPSG:3006.")
             parishes_gdf.set_crs("EPSG:3006", inplace=True)
        logging.info(f"Loaded parishes shapefile with CRS: {parishes_gdf.crs}")
        # Verify required columns exist (using corrected names)
        required_cols = ['prsh_cd', 'prsh_nm', 'wstrn_l', 'geometry']
        if not all(col in parishes_gdf.columns for col in required_cols):
             logging.error(f"Parish shapefile missing required columns. Need: {required_cols}. Found: {list(parishes_gdf.columns)}")
             return 1
    except Exception as e:
        logging.error(f"Failed to load or verify parish shapefile: {e}")
        return 1

    # --- 3. Consolidate and Map Unique Locations ---
    unique_locations = set()
    for person in persons_data_list:
        # Add birth location
        if pd.notna(person.get('birth_location_lat')) and pd.notna(person.get('birth_location_lon')):
            unique_locations.add((person['birth_location_lat'], person['birth_location_lon']))
        # Add current location
        if pd.notna(person.get('current_location_lat')) and pd.notna(person.get('current_location_lon')):
            unique_locations.add((person['current_location_lat'], person['current_location_lon']))
        # Add career locations
        for job in person.get('_career_raw', []):
            loc = get_location_details(job.get('location')) # Use helper to handle extraction and validation
            if pd.notna(loc['lat']) and pd.notna(loc['lon']):
                unique_locations.add((loc['lat'], loc['lon']))

    unique_locations_df = pd.DataFrame(list(unique_locations), columns=['lat', 'lon'])
    
    # Perform the mapping for all unique points found
    location_parish_lookup = map_unique_locations(unique_locations_df, parishes_gdf)

    # --- 4. Assign Mapped Info and Calculate Career Flags ---
    logging.info("Assigning mapped parish info and calculating career flags...")
    processed_persons_list = []
    for person in persons_data_list:
        # Initialize new fields
        person['birth_parish_code'] = None
        person['birth_parish_name'] = None
        person['birth_parish_is_western_line'] = -1 # Use -1 for unmapped/unknown
        person['currently_lives_in_wl'] = -1
        person['worked_wl_before_1930'] = False # Initialize flags to False
        person['worked_wl_after_1930'] = False

        # Assign birth parish info
        birth_coords = (person.get('birth_location_lat'), person.get('birth_location_lon'))
        if birth_coords in location_parish_lookup:
            parish_info = location_parish_lookup[birth_coords]
            person['birth_parish_code'] = parish_info['parish_code']
            person['birth_parish_name'] = parish_info['parish_name']
            person['birth_parish_is_western_line'] = parish_info['western_line']

        # Assign current location info
        current_coords = (person.get('current_location_lat'), person.get('current_location_lon'))
        if current_coords in location_parish_lookup:
            parish_info = location_parish_lookup[current_coords]
            person['currently_lives_in_wl'] = parish_info['western_line']

        # Process career locations
        for job in person.get('_career_raw', []):
            loc = get_location_details(job.get('location'))
            job_coords = (loc['lat'], loc['lon'])
            
            if job_coords in location_parish_lookup:
                parish_info = location_parish_lookup[job_coords]
                # Check if job location is in a Western Line parish
                if parish_info['western_line'] == 1: 
                    start_year = pd.to_numeric(job.get('start_year'), errors='coerce')
                    end_year = pd.to_numeric(job.get('end_year'), errors='coerce')
                    
                    # Determine if job overlaps with pre/post 1930 periods
                    # Note: This logic assumes job spans entire start-end year. Adjust if needed.
                    # Handles missing end_year by assuming job continues indefinitely or at least past 1930.
                    
                    # Check pre-1930
                    if pd.notna(start_year) and start_year < 1930:
                         person['worked_wl_before_1930'] = True
                    
                    # Check post-1930 (start year is >= 1930 OR end year is >= 1930 OR end year is missing and start is valid)
                    if pd.notna(start_year):
                        if start_year >= 1930:
                             person['worked_wl_after_1930'] = True
                        elif pd.notna(end_year) and end_year >= 1930:
                             person['worked_wl_after_1930'] = True
                        elif pd.isna(end_year): # If end year is missing, assume it potentially runs past 1930
                             person['worked_wl_after_1930'] = True
                             
        # Remove temporary raw fields before creating final DataFrame
        person.pop('_education_raw', None)
        person.pop('_career_raw', None)
        processed_persons_list.append(person)
        
    # --- 5. Create and Save Final DataFrame ---
    final_df = pd.DataFrame(processed_persons_list)
    
    # Convert boolean flags explicitly
    final_df['worked_wl_before_1930'] = final_df['worked_wl_before_1930'].astype(bool)
    final_df['worked_wl_after_1930'] = final_df['worked_wl_after_1930'].astype(bool)
    # Ensure WL flags are integers (-1, 0, 1)
    final_df['birth_parish_is_western_line'] = final_df['birth_parish_is_western_line'].astype(int)
    final_df['currently_lives_in_wl'] = final_df['currently_lives_in_wl'].astype(int)


    output_csv_path = os.path.join(OUTPUT_DIR, 'persons_data_final_mapped.csv')
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logging.info(f"Saved final mapped data for {len(final_df)} persons to {output_csv_path}")
    
    # --- Optional: Run other analyses ---
    # ... (decade analysis, metadata extraction using final_df.to_dict('records') if needed) ...
    
    logging.info("Processing finished.")
    return 0

if __name__ == "__main__":
    try:
        import geopandas
    except ImportError:
        logging.error("Module 'geopandas' not found. Please install it (`pip install geopandas`)")
        exit(1) 
    main()