import os
import json
import pandas as pd
import geopandas as gpd # Required external library
from glob import glob
import logging
from collections import defaultdict # Kept import just in case, though Counter might be more specific if needed elsewhere
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import numpy as np # Required for NaN handling

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
# Define directories & Paths (User should verify these)
INPUT_DIR: str = "data/enriched_biographies"
OUTPUT_DIR: str = "data/analysis"
PARISH_SHAPEFILE_PATH: str = "data/parishes/parish_map_1920.shp"
# **** NEW: Path to your institution mapping JSON file ****
INSTITUTION_MAPPING_PATH: str = "data/analysis/institution_mapping.json" 
# Updated output file name to reflect standardized education
FINAL_CSV_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_final_mapped_std_edu.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HISCO code ranges
ENGINEER_HISCO_RANGES: List[Tuple[int, int]] = [(1000, 3999)]
DIRECTOR_HISCO_RANGES: List[Tuple[int, int]] = [(21000, 21900)]
RELEVANT_HISCO_RANGES: List[Tuple[int, int]] = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES

# Career analysis threshold
CAREER_YEAR_THRESHOLD: int = 1930

# Shapefile column names (Using corrected names based on user feedback)
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
    hisco_code = occupation.get("hisco_code_swedish") or occupation.get("hisco_code_english")
    return check_hisco_range(hisco_code, RELEVANT_HISCO_RANGES)

def extract_birth_decade(birth_date: Optional[str]) -> Optional[int]:
    """Extract the birth decade (e.g., 1910) from various date string formats."""
    if not birth_date or not isinstance(birth_date, str): return None
    year: Optional[int] = None
    year_match = re.search(r'(?:^|\D)(\d{4})(?:$|\D)', birth_date)
    if year_match:
        try: year = int(year_match.group(1))
        except ValueError: pass
    if year and 1700 < year < 2020: return (year // 10) * 10
    return None

def classify_education(education_entries: Optional[List[Dict[str, Any]]]) -> Dict[str, bool]:
    """Classify education into technical, business, or other higher education."""
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
        is_higher: bool = entry.get("degree_level") in HIGHER_EDU_LEVELS or any(keyword in full_text for keyword in HIGHER_EDU_KW)
        if is_higher: has_higher_edu = True
        if any(keyword in full_text for keyword in TECH_KW): result["technical"] = True
        elif any(keyword in full_text for keyword in BIZ_KW): result["business"] = True
        elif is_higher: result["other_higher"] = True
    if not has_higher_edu: result["other_higher"] = False
    return result

def get_location_details(location_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely extract and validate location coordinates."""
    if not location_data or not isinstance(location_data, dict): return {"name": None, "lat": None, "lon": None}
    lat = pd.to_numeric(location_data.get("latitude"), errors='coerce')
    lon = pd.to_numeric(location_data.get("longitude"), errors='coerce')
    lat_valid = pd.notna(lat) and -90 <= lat <= 90
    lon_valid = pd.notna(lon) and -180 <= lon <= 180
    return {"name": location_data.get("name"), "lat": lat if lat_valid else None, "lon": lon if lon_valid else None}

def get_overseas_experience(career_entries: Optional[List[Dict[str, Any]]]) -> Tuple[bool, bool, List[str]]:
    """Identify overseas and US experience from career list."""
    has_overseas, has_us, countries = False, False, set()
    if not career_entries: return has_overseas, has_us, []
    for entry in career_entries:
        country_code = entry.get("country_code")
        if country_code and isinstance(country_code, str) and country_code.upper() != "SWE":
            has_overseas = True
            cc_upper = country_code.upper()
            countries.add(cc_upper)
            if cc_upper == "USA": has_us = True
    return has_overseas, has_us, sorted(list(countries))

# --- **** NEW: Function to Standardize Education Institutions **** ---
def standardize_education_data(education_list_json: str, mapping: Dict[str, str]) -> str:
    """
    Parses a JSON string containing education entries, standardizes 
    institution names using a mapping, and returns a JSON string of the results.

    Args:
        education_list_json: JSON string of the list of education dictionaries.
        mapping: The dictionary mapping original names to standardized names.

    Returns:
        A JSON string of the list of education dictionaries with added 
        'institution_standardized' field. Returns original JSON string on error.
    """
    try:
        education_list = json.loads(education_list_json)
        if not isinstance(education_list, list): return education_list_json # Return original if not list
    except json.JSONDecodeError:
        logging.warning(f"Could not parse education JSON string for standardization: {education_list_json[:100]}...")
        return education_list_json # Return original on parse error

    standardized_list = []
    for entry in education_list:
        new_entry = entry.copy() 
        original_name = new_entry.get("institution")
        
        if original_name and isinstance(original_name, str):
            normalized_original = original_name.strip()
            # Map name, default to original if not found in mapping
            standardized_name = mapping.get(normalized_original, normalized_original) 
            new_entry["institution_standardized"] = standardized_name
        else:
            # Assign a default standardized name if original is missing/invalid
            new_entry["institution_standardized"] = mapping.get("Unknown/Unspecified", "Unknown/Unspecified") 
            
        standardized_list.append(new_entry)
        
    # Return the modified list as a JSON string
    return json.dumps(standardized_list, ensure_ascii=False)


# --- Data Extraction Function (from user's provided 'working' script) ---
# Note: This now saves _education_raw and _career_raw as JSON strings directly
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
            if not person or not is_relevant_person(person): continue 

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

            person_record: Dict[str, Any] = {
                'person_id': os.path.basename(filename).replace('.json', ''),
                'person_name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                'birth_date': person.get('birth_date'),
                'birth_decade': extract_birth_decade(person.get('birth_date')),
                'birth_location_name': birth_loc["name"], 'birth_location_lat': birth_loc["lat"], 'birth_location_lon': birth_loc["lon"], 
                'current_location_name': current_loc["name"], 'current_location_lat': current_loc["lat"], 'current_location_lon': current_loc["lon"],
                'person_occupation': occupation.get('occupation'), 'person_hisco_swe': occupation.get('hisco_code_swedish'), 'person_hisco_eng': occupation.get('hisco_code_english'),
                'father_name': father.get('name'), 'father_occupation': father.get('occupation'), 'father_hisco': father.get('hisco_code'),
                'edu_technical': education_classification['technical'], 'edu_business': education_classification['business'], 'edu_other_higher': education_classification['other_higher'],
                'career_has_overseas': has_overseas, 'career_has_us': has_us, 'career_overseas_countries': ",".join(overseas_countries), 
                'board_membership_count': board_count,
                # **** Store raw data as JSON STRINGS ****
                '_education_raw': json.dumps(education_entries, ensure_ascii=False), 
                '_career_raw': json.dumps(career_entries, ensure_ascii=False),
             }
            persons_data_list.append(person_record)
        except Exception as e: logging.error(f"Error processing file {os.path.basename(filename)}: {e}", exc_info=False)

    logging.info(f"Completed initial data extraction. Found {len(persons_data_list)} relevant persons.")
    return persons_data_list


# --- Function to Map Unique Locations (from 'working' script version) ---
def map_unique_locations(locations_df: pd.DataFrame, parishes_gdf: gpd.GeoDataFrame) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Maps unique lat/lon pairs to parish information using spatial join."""
    shp_cols: List[str] = [SHP_GEOMETRY_COL, SHP_PARISH_CODE_COL, SHP_PARISH_NAME_COL, SHP_WESTERN_LINE_COL]
    if locations_df.empty: return {}
    logging.info(f"Mapping {len(locations_df)} unique locations to parishes...")
    location_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
    try:
        locations_gdf = gpd.GeoDataFrame(locations_df, geometry=gpd.points_from_xy(locations_df['lon'], locations_df['lat']), crs="EPSG:4326")
        locations_gdf = locations_gdf.to_crs(parishes_gdf.crs)
        parishes_simplified = parishes_gdf[shp_cols].copy()
        joined_gdf = gpd.sjoin(locations_gdf, parishes_simplified, how='left', predicate='within')
        
        for original_idx, row in joined_gdf.iterrows():
            lat, lon = locations_df.loc[original_idx, 'lat'], locations_df.loc[original_idx, 'lon']
            wl_status_raw = row.get(SHP_WESTERN_LINE_COL)
            wl_status: Optional[bool] = None # Standardize to True/False/None
            if pd.notna(wl_status_raw):
                 try: wl_status = bool(int(wl_status_raw))
                 except (ValueError, TypeError): pass 
            location_map[(lat, lon)] = {
                'parish_code': row.get(SHP_PARISH_CODE_COL), 
                'parish_name': row.get(SHP_PARISH_NAME_COL), 
                'is_western_line': wl_status # Store standardized boolean or None
            }
        logging.info(f"Finished mapping unique locations.")
    except Exception as e:
        logging.error(f"Error during unique location mapping: {e}", exc_info=True)
        return {} 
    return location_map


# --- Main Execution ---
def main() -> int:
    """Main execution function."""
    logging.info(f"Starting data extraction, mapping, and standardization...")
    
    # --- 1. Load Institution Mapping ---
    inst_mapping: Dict[str, str] = {}
    if not os.path.exists(INSTITUTION_MAPPING_PATH):
        logging.warning(f"Institution mapping file not found: {INSTITUTION_MAPPING_PATH}. Education institutions will not be standardized.")
    else:
        try:
            # Explicitly load with utf-8, json handles unicode escapes
            with open(INSTITUTION_MAPPING_PATH, 'r', encoding='utf-8') as f: 
                inst_mapping = json.load(f)
            logging.info(f"Loaded institution mapping ({len(inst_mapping)} entries).")
        except Exception as e:
            logging.error(f"Failed to load institution mapping: {e}. Proceeding without standardization.")
            inst_mapping = {} 
            
    # --- 2. Extract Raw Data (yields list of dicts) ---
    persons_data_list: List[Dict[str, Any]] = extract_persons_data()
    if not persons_data_list: return 1
        
    # --- 3. Load and Verify Parish Shapefile ---
    # (Same as previous version - ensures parishes_gdf is loaded and valid)
    if not os.path.exists(PARISH_SHAPEFILE_PATH):
        logging.error(f"Parish shapefile not found: {PARISH_SHAPEFILE_PATH}.")
        return 1
    try:
        parishes_gdf = gpd.read_file(PARISH_SHAPEFILE_PATH)
        if not parishes_gdf.crs: parishes_gdf.set_crs("EPSG:3006", inplace=True)
        logging.info(f"Loaded parishes shapefile (CRS: {parishes_gdf.crs})")
        required_shp_cols: List[str] = [SHP_PARISH_CODE_COL, SHP_PARISH_NAME_COL, SHP_WESTERN_LINE_COL, SHP_GEOMETRY_COL]
        if not all(col in parishes_gdf.columns for col in required_shp_cols):
             logging.error(f"Parish shapefile missing required columns. Need: {required_shp_cols}.")
             return 1
    except Exception as e:
        logging.error(f"Failed to load parish shapefile: {e}")
        return 1

    # --- 4. Consolidate and Map Unique Locations ---
    # (Same as previous version - creates location_parish_lookup dict)
    unique_locations: Set[Tuple[float, float]] = set()
    for person in persons_data_list:
        for loc_type in ['birth_location', 'current_location']:
            lat, lon = person.get(f"{loc_type}_lat"), person.get(f"{loc_type}_lon")
            if pd.notna(lat) and pd.notna(lon): unique_locations.add((lat, lon))
        # Parse career JSON string to get locations
        try:
            career_list = json.loads(person.get('_career_raw', '[]'))
            for job in career_list:
                 loc = get_location_details(job.get('location'))
                 if pd.notna(loc['lat']) and pd.notna(loc['lon']): unique_locations.add((loc['lat'], loc['lon']))
        except json.JSONDecodeError: pass # Ignore if career raw is not valid JSON
        # Parse education JSON string to get locations
        try:
            edu_list = json.loads(person.get('_education_raw', '[]'))
            for edu in edu_list:
                 # Assuming location is nested directly in edu entry
                 loc = get_location_details(edu.get('location')) 
                 if pd.notna(loc['lat']) and pd.notna(loc['lon']): unique_locations.add((loc['lat'], loc['lon']))
        except json.JSONDecodeError: pass # Ignore if education raw is not valid JSON
            
    if not unique_locations: location_parish_lookup = {}
    else:
         unique_locations_df = pd.DataFrame(list(unique_locations), columns=['lat', 'lon'])
         location_parish_lookup = map_unique_locations(unique_locations_df, parishes_gdf)

    # --- 5. Process Each Person: Assign Mapped Info, Standardize Edu, Calculate Flags ---
    logging.info("Assigning mapped info, standardizing education, calculating flags...")
    processed_persons_list: List[Dict[str, Any]] = []
    for person in persons_data_list: # person is a dict
        # --- Assign Birth/Current Location Info ---
        birth_coords = (person.get('birth_location_lat'), person.get('birth_location_lon'))
        birth_parish_info = location_parish_lookup.get(birth_coords, {}) 
        person['birth_parish_code'] = birth_parish_info.get('parish_code')
        person['birth_parish_name'] = birth_parish_info.get('parish_name')
        person['birth_parish_is_western_line'] = birth_parish_info.get('is_western_line') # True/False/None

        current_coords = (person.get('current_location_lat'), person.get('current_location_lon'))
        current_parish_info = location_parish_lookup.get(current_coords, {})
        person['currently_lives_in_wl'] = current_parish_info.get('is_western_line') # True/False/None
        
        # --- Calculate Career Flags ---
        worked_wl_before = False
        worked_wl_after = False
        try:
            # Parse career JSON string
            raw_career_list = json.loads(person.get('_career_raw', '[]')) 
            if isinstance(raw_career_list, list): # Check if parsing succeeded
                for job in raw_career_list:
                    loc = get_location_details(job.get('location'))
                    job_coords = (loc['lat'], loc['lon'])
                    job_parish_info = location_parish_lookup.get(job_coords)
                    
                    if job_parish_info and job_parish_info.get('is_western_line') is True:
                        start_year = pd.to_numeric(job.get('start_year'), errors='coerce')
                        end_year = pd.to_numeric(job.get('end_year'), errors='coerce')
                        if pd.notna(start_year):
                            if start_year < CAREER_YEAR_THRESHOLD: worked_wl_before = True
                            if start_year >= CAREER_YEAR_THRESHOLD or \
                               (pd.notna(end_year) and end_year >= CAREER_YEAR_THRESHOLD) or \
                               pd.isna(end_year): worked_wl_after = True
        except json.JSONDecodeError:
             logging.warning(f"Could not parse _career_raw for person {person.get('person_id')}")
                           
        person['worked_wl_before_1930'] = worked_wl_before
        person['worked_wl_after_1930'] = worked_wl_after
        
        # --- **** NEW: Standardize Education **** ---
        # Standardize based on the _education_raw JSON string
        if inst_mapping: # Only run if mapping exists
            person['education_standardized'] = standardize_education_data(
                person.get('_education_raw', '[]'), 
                inst_mapping
            )
        else:
            # If no mapping, just keep the original JSON string here too
            person['education_standardized'] = person.get('_education_raw', '[]')

        # Note: _education_raw and _career_raw remain as original JSON strings
        
        processed_persons_list.append(person)
        
    # --- 6. Create and Save Final DataFrame ---
    final_df = pd.DataFrame(processed_persons_list)
    
    # Set data types for clarity (optional, pandas usually infers well)
    # Boolean types
    bool_cols = ['edu_technical', 'edu_business', 'edu_other_higher', 
                 'career_has_overseas', 'career_has_us',
                 'worked_wl_before_1930', 'worked_wl_after_1930']
    for col in bool_cols:
        if col in final_df.columns: final_df[col] = final_df[col].astype(bool)
    # Nullable Boolean (object type handles None/NA)
    nullable_bool_cols = ['birth_parish_is_western_line', 'currently_lives_in_wl']
    # No explicit conversion needed if None should be preserved along with True/False

    logging.info("Final DataFrame columns and dtypes:\n%s", final_df.info())
    
    try:
        # Save with utf-8-sig encoding
        final_df.to_csv(FINAL_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig') 
        logging.info(f"Saved final mapped & standardized data ({len(final_df)} rows) to: {FINAL_CSV_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save final CSV: {e}")
        return 1
        
    logging.info("Processing finished successfully.")
    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        import geopandas
        logging.info(f"Geopandas version: {geopandas.__version__}")
    except ImportError:
        logging.error("Required module 'geopandas' not found. Please install it.")
        exit(1) 
    main()