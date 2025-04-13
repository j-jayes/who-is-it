import os
import json
import pandas as pd
import geopandas as gpd # Required external library
from glob import glob
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import numpy as np # Required for NaN handling

# --- Basic Setup ---
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories & Paths (User should verify these)
INPUT_DIR: str = "data/enriched_biographies"
OUTPUT_DIR: str = "data/analysis"
PARISH_SHAPEFILE_PATH: str = "data/parishes/parish_map_1920.shp"
FINAL_CSV_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_final_mapped.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
# HISCO code ranges
ENGINEER_HISCO_RANGES: List[Tuple[int, int]] = [(1000, 3999)]
DIRECTOR_HISCO_RANGES: List[Tuple[int, int]] = [(21000, 21900)]
RELEVANT_HISCO_RANGES: List[Tuple[int, int]] = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES

# Career analysis threshold
CAREER_YEAR_THRESHOLD: int = 1930

# Shapefile column names (User should verify these match their shapefile)
SHP_PARISH_CODE_COL: str = 'prsh_cd'
SHP_PARISH_NAME_COL: str = 'prsh_nm'
SHP_WESTERN_LINE_COL: str = 'wstrn_l'
SHP_GEOMETRY_COL: str = 'geometry'

# --- Helper Functions ---

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
    # Prioritize Swedish code, fallback to English
    hisco_code = occupation.get("hisco_code_swedish") or occupation.get("hisco_code_english")
    return check_hisco_range(hisco_code, RELEVANT_HISCO_RANGES)

def extract_birth_decade(birth_date: Optional[str]) -> Optional[int]:
    """Extract the birth decade (e.g., 1910) from various date string formats."""
    if not birth_date or not isinstance(birth_date, str): return None
    year: Optional[int] = None
    # Match YYYY at the start or end, or in YYYY-MM-DD, DD-MM-YYYY
    year_match = re.search(r'(?:^|\D)(\d{4})(?:$|\D)', birth_date)
    if year_match:
        try: year = int(year_match.group(1))
        except ValueError: pass
        
    if year and 1700 < year < 2020: # Added upper bound sanity check
         return (year // 10) * 10
    elif year:
         logging.debug(f"Extracted year {year} out of plausible range from: {birth_date}")
    return None # Return None if no valid year found

def classify_education(education_entries: Optional[List[Dict[str, Any]]]) -> Dict[str, bool]:
    """Classify education into technical, business, or other higher education."""
    # Simplified keywords - User may want to expand these lists
    TECH_KW: List[str] = ['tekniska', 'chalmers', 'kth', 'tekn', 'ingenjör', 'teknisk', 'teknolog', 'polytekn', 'engineering', 'technical']
    BIZ_KW: List[str] = ['handels', 'ekonomi', 'handelshögskola', 'business', 'commerce', 'ekonomisk', 'handelsinstitut', 'handelsgymnasium', 'ekonom']
    HIGHER_EDU_LEVELS: List[str] = ["Master's", "PhD", "Bachelor's", "Licentiate"]
    HIGHER_EDU_KW: List[str] = ['högskola', 'universitet', 'university', 'college']

    if not education_entries: return {"technical": False, "business": False, "other_higher": False}
    
    result: Dict[str, bool] = {"technical": False, "business": False, "other_higher": False}
    has_higher_edu: bool = False
    
    for entry in education_entries:
        institution: str = str(entry.get("institution", "")).lower()
        degree: str = str(entry.get("degree", "")).lower()
        full_text: str = institution + " " + degree
        
        is_higher: bool = entry.get("degree_level") in HIGHER_EDU_LEVELS or \
                          any(keyword in full_text for keyword in HIGHER_EDU_KW)
        if is_higher: has_higher_edu = True
        
        # Use elif to avoid double counting if keywords overlap
        if any(keyword in full_text for keyword in TECH_KW): result["technical"] = True
        elif any(keyword in full_text for keyword in BIZ_KW): result["business"] = True
        elif is_higher: result["other_higher"] = True # Only flag if not technical/business

    if not has_higher_edu: result["other_higher"] = False # Ensure flag is False if no higher edu found
    return result

def get_location_details(location_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely extract and validate location coordinates."""
    if not location_data or not isinstance(location_data, dict):
        return {"name": None, "lat": None, "lon": None} # Removed formatted address for simplicity
        
    lat = pd.to_numeric(location_data.get("latitude"), errors='coerce')
    lon = pd.to_numeric(location_data.get("longitude"), errors='coerce')
    # Basic validity check for lat/lon ranges
    lat_valid = pd.notna(lat) and -90 <= lat <= 90
    lon_valid = pd.notna(lon) and -180 <= lon <= 180
    
    return {
        "name": location_data.get("name"),
        "lat": lat if lat_valid else None,
        "lon": lon if lon_valid else None,
    }

def get_overseas_experience(career_entries: Optional[List[Dict[str, Any]]]) -> Tuple[bool, bool, List[str]]:
    """Identify overseas and US experience from career list."""
    has_overseas, has_us, countries = False, False, set()
    if not career_entries: return has_overseas, has_us, []
    
    for entry in career_entries:
        country_code = entry.get("country_code")
        # Check if country code exists and is not Sweden (case-insensitive)
        if country_code and isinstance(country_code, str) and country_code.upper() != "SWE":
            has_overseas = True
            cc_upper = country_code.upper()
            countries.add(cc_upper)
            if cc_upper == "USA": has_us = True
            
    return has_overseas, has_us, sorted(list(countries))

# --- Data Extraction Function ---
def extract_persons_data() -> List[Dict[str, Any]]:
    """Extract key information from JSON files into a list of dictionaries."""
    persons_data_list: List[Dict[str, Any]] = []
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return persons_data_list

    all_files: List[str] = glob(os.path.join(INPUT_DIR, "*.json"))
    logging.info(f"Found {len(all_files)} potential JSON files in {INPUT_DIR}")
    files_processed: int = 0
    
    for filename in all_files:
        files_processed += 1
        if files_processed % 500 == 0: logging.info(f"Processed {files_processed}/{len(all_files)} files...")

        try:
            with open(filename, 'r', encoding='utf-8') as f: data = json.load(f)
            person = data.get('person', {})
            # Skip if no person data or person is not relevant based on occupation
            if not person or not is_relevant_person(person): continue 

            # --- Extract Details ---
            birth_loc = get_location_details(person.get('birth_place'))
            current_loc = get_location_details(data.get('current_location'))
            occupation = person.get('occupation', {}) or {}
            
            father: Dict[str, Any] = {}
            parents = person.get('parents', [])
            if parents:
                 for p in parents:
                     if p and p.get('gender') == 'Male':
                         father = {'name': f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                                   'occupation': (p.get('occupation') or {}).get('occupation'),
                                   'hisco_code': (p.get('occupation') or {}).get('hisco_code_swedish') or (p.get('occupation') or {}).get('hisco_code_english') }
                         break
                         
            education_entries: List[Dict] = data.get('education', []) or []
            education_classification: Dict[str, bool] = classify_education(education_entries)
            career_entries: List[Dict] = data.get('career', []) or []
            has_overseas, has_us, overseas_countries = get_overseas_experience(career_entries)
            board_count: int = len(data.get('board_memberships', []) or [])

            # --- Create Record ---
            person_record: Dict[str, Any] = {
                'person_id': os.path.basename(filename).replace('.json', ''),
                'person_name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                'birth_date': person.get('birth_date'),
                'birth_decade': extract_birth_decade(person.get('birth_date')),
                'birth_location_name': birth_loc["name"],
                'birth_location_lat': birth_loc["lat"], 
                'birth_location_lon': birth_loc["lon"], 
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
                # Keep raw data as complex types for now
                '_education_raw': education_entries, 
                '_career_raw': career_entries,
             }
            persons_data_list.append(person_record)
        except Exception as e: logging.error(f"Error processing file {os.path.basename(filename)}: {e}", exc_info=False)

    logging.info(f"Completed initial data extraction. Found {len(persons_data_list)} relevant persons.")
    return persons_data_list


# --- Function to Map Unique Locations ---
def map_unique_locations(locations_df: pd.DataFrame, parishes_gdf: gpd.GeoDataFrame) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Maps unique lat/lon pairs to parish information using spatial join."""
    # Use correct shapefile column names defined globally
    shp_cols: List[str] = [SHP_GEOMETRY_COL, SHP_PARISH_CODE_COL, SHP_PARISH_NAME_COL, SHP_WESTERN_LINE_COL]
    
    if locations_df.empty: return {}
    logging.info(f"Mapping {len(locations_df)} unique locations to parishes...")
    location_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
    
    try:
        locations_gdf = gpd.GeoDataFrame(
            locations_df, geometry=gpd.points_from_xy(locations_df['lon'], locations_df['lat']), crs="EPSG:4326"
        )
        locations_gdf = locations_gdf.to_crs(parishes_gdf.crs) # Align CRS
        
        parishes_simplified = parishes_gdf[shp_cols].copy()
        joined_gdf = gpd.sjoin(locations_gdf, parishes_simplified, how='left', predicate='within')

        # Create lookup dictionary mapping (lat, lon) to parish info
        for original_idx, row in joined_gdf.iterrows():
            # Use original_idx (which is the index from locations_df) to get lat/lon
            lat, lon = locations_df.loc[original_idx, 'lat'], locations_df.loc[original_idx, 'lon']
            wl_status_raw = row.get(SHP_WESTERN_LINE_COL) # Get raw value (could be 0, 1, NaN)
            
            # Standardize western_line status to True/False/None
            wl_status: Optional[bool] = None
            if pd.notna(wl_status_raw):
                 try: 
                      wl_status = bool(int(wl_status_raw)) # Convert 1 to True, 0 to False
                 except (ValueError, TypeError): pass # Keep None if conversion fails

            location_map[(lat, lon)] = {
                'parish_code': row.get(SHP_PARISH_CODE_COL), 
                'parish_name': row.get(SHP_PARISH_NAME_COL), 
                'is_western_line': wl_status # Store standardized boolean or None
            }
            
        logging.info(f"Finished mapping unique locations. Created lookup for {len(location_map)} coordinates.")
    except Exception as e:
        logging.error(f"Error during unique location mapping: {e}", exc_info=True)
        return {} 
    return location_map


# --- Main Execution ---
def main() -> int:
    """Main execution function: Extracts, maps, processes, and saves data."""
    logging.info(f"Starting data extraction and mapping process...")
    
    # --- 1. Extract Raw Data ---
    persons_data_list: List[Dict[str, Any]] = extract_persons_data()
    if not persons_data_list:
        logging.warning("No relevant persons found. Exiting.")
        return 1
        
    # --- 2. Load and Verify Parish Shapefile ---
    if not os.path.exists(PARISH_SHAPEFILE_PATH):
        logging.error(f"Parish shapefile not found: {PARISH_SHAPEFILE_PATH}. Cannot perform spatial mapping.")
        return 1
        
    try:
        parishes_gdf = gpd.read_file(PARISH_SHAPEFILE_PATH)
        if not parishes_gdf.crs: # Set CRS if missing
             logging.warning(f"Parish shapefile CRS missing. Assuming EPSG:3006 based on standard Swedish projection.")
             parishes_gdf.set_crs("EPSG:3006", inplace=True)
        logging.info(f"Loaded parishes shapefile (CRS: {parishes_gdf.crs})")
        # Verify required columns exist (using defined constants)
        required_shp_cols: List[str] = [SHP_PARISH_CODE_COL, SHP_PARISH_NAME_COL, SHP_WESTERN_LINE_COL, SHP_GEOMETRY_COL]
        if not all(col in parishes_gdf.columns for col in required_shp_cols):
             logging.error(f"Parish shapefile missing required columns. Need: {required_shp_cols}. Found: {list(parishes_gdf.columns)}")
             return 1
    except Exception as e:
        logging.error(f"Failed to load or verify parish shapefile: {e}")
        return 1

    # --- 3. Consolidate and Map Unique Locations ---
    unique_locations: Set[Tuple[float, float]] = set()
    for person in persons_data_list:
        # Consolidate all valid lat/lon pairs
        for loc_type in ['birth_location', 'current_location']:
            lat, lon = person.get(f"{loc_type}_lat"), person.get(f"{loc_type}_lon")
            if pd.notna(lat) and pd.notna(lon): unique_locations.add((lat, lon))
        for job in person.get('_career_raw', []):
            loc = get_location_details(job.get('location'))
            if pd.notna(loc['lat']) and pd.notna(loc['lon']): unique_locations.add((loc['lat'], loc['lon']))
            
    if not unique_locations:
         logging.warning("No valid coordinates found in any records to map.")
         location_parish_lookup = {}
    else:
         unique_locations_df = pd.DataFrame(list(unique_locations), columns=['lat', 'lon'])
         location_parish_lookup = map_unique_locations(unique_locations_df, parishes_gdf)

    # --- 4. Assign Mapped Info and Calculate Career Flags ---
    logging.info("Assigning mapped parish info and calculating flags...")
    processed_persons_list: List[Dict[str, Any]] = []
    for person in persons_data_list:
        # Look up parish info for birth location
        birth_coords = (person.get('birth_location_lat'), person.get('birth_location_lon'))
        birth_parish_info = location_parish_lookup.get(birth_coords, {}) # Get info or empty dict
        person['birth_parish_code'] = birth_parish_info.get('parish_code')
        person['birth_parish_name'] = birth_parish_info.get('parish_name')
        person['birth_parish_is_western_line'] = birth_parish_info.get('is_western_line') # Already True/False/None

        # Look up parish info for current location
        current_coords = (person.get('current_location_lat'), person.get('current_location_lon'))
        current_parish_info = location_parish_lookup.get(current_coords, {})
        person['currently_lives_in_wl'] = current_parish_info.get('is_western_line') # Already True/False/None
        
        # Calculate career flags
        worked_wl_before = False
        worked_wl_after = False
        for job in person.get('_career_raw', []):
            loc = get_location_details(job.get('location'))
            job_coords = (loc['lat'], loc['lon'])
            job_parish_info = location_parish_lookup.get(job_coords)
            
            # Check if worked in a WL parish
            if job_parish_info and job_parish_info.get('is_western_line') is True:
                start_year = pd.to_numeric(job.get('start_year'), errors='coerce')
                end_year = pd.to_numeric(job.get('end_year'), errors='coerce')
                
                if pd.notna(start_year):
                    # Check pre-threshold: If job started before threshold
                    if start_year < CAREER_YEAR_THRESHOLD:
                        worked_wl_before = True
                    # Check post-threshold: If job started at/after threshold OR ended at/after threshold OR end is missing
                    if start_year >= CAREER_YEAR_THRESHOLD or \
                       (pd.notna(end_year) and end_year >= CAREER_YEAR_THRESHOLD) or \
                       pd.isna(end_year):
                           worked_wl_after = True
                           
        person['worked_wl_before_1930'] = worked_wl_before
        person['worked_wl_after_1930'] = worked_wl_after
        
        # Convert raw data to JSON strings for CSV compatibility
        person['_education_raw'] = json.dumps(person.get('_education_raw', []))
        person['_career_raw'] = json.dumps(person.get('_career_raw', []))

        processed_persons_list.append(person)
        
    # --- 5. Create and Save Final DataFrame ---
    final_df = pd.DataFrame(processed_persons_list)
    
    # Final check/conversion for boolean/nullable boolean types if needed
    # Pandas usually handles True/False/None well, but explicit checks can be added
    logging.info("Final DataFrame columns and dtypes:\n%s", final_df.info())
    
    try:
        final_df.to_csv(FINAL_CSV_OUTPUT_PATH, index=False, encoding='utf-8')
        logging.info(f"Saved final mapped data ({len(final_df)} rows) to: {FINAL_CSV_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save final CSV: {e}")
        return 1
        
    logging.info("Processing finished successfully.")
    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    # Check for geopandas dependency before running main
    try:
        import geopandas
        logging.info(f"Geopandas version: {geopandas.__version__}")
    except ImportError:
        logging.error("Required module 'geopandas' not found.")
        logging.error("Please install it, e.g., 'pip install geopandas' or 'conda install geopandas'.")
        exit(1) 
        
    exit_code = main()
    exit(exit_code)