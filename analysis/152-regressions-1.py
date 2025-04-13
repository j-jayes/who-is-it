import os
import json
import pandas as pd
import geopandas as gpd # Required external library
import statsmodels.formula.api as smf # Required external library
from glob import glob
import logging
from collections import defaultdict 
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import numpy as np # Required external library

# --- Basic Setup ---
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration ---
# Define directories & Paths (User should verify these)
INPUT_DIR: str = "data/enriched_biographies"
OUTPUT_DIR: str = "data/analysis"
PARISH_SHAPEFILE_PATH: str = "data/parishes/parish_map_1920.shp"
INSTITUTION_MAPPING_PATH: str = os.path.join(OUTPUT_DIR, "institution_mapping.json") 
FINAL_CSV_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_final_for_regression.csv') # Final output name

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HISCO code ranges
ENGINEER_HISCO_RANGES: List[Tuple[int, int]] = [(1000, 3999)] 
DIRECTOR_HISCO_RANGES: List[Tuple[int, int]] = [(21000, 21900)] 
RELEVANT_HISCO_RANGES: List[Tuple[int, int]] = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES

# Career analysis threshold
CAREER_YEAR_THRESHOLD: int = 1930

# Shapefile column names (Ensure these match your shapefile)
SHP_PARISH_CODE_COL: str = 'prsh_cd'
SHP_PARISH_NAME_COL: str = 'prsh_nm'
SHP_WESTERN_LINE_COL: str = 'wstrn_l'
SHP_GEOMETRY_COL: str = 'geometry'

# Network calculation parameters
COHORT_YEAR_WINDOW: int = 4 # Set to +/- 4 years per user request


# --- Helper Functions ---

def safe_float_to_int(value: Any) -> Optional[int]:
    """Safely convert a value to integer."""
    if value is None: return None
    try: return int(float(value))
    except (ValueError, TypeError): return None

def check_hisco_range(code_str: Optional[str], ranges: List[Tuple[int, int]]) -> bool:
    """Check HISCO code against ranges."""
    code_value = safe_float_to_int(code_str)
    if code_value is None: return False
    for min_val, max_val in ranges:
        if min_val <= code_value <= max_val: return True
    return False

def is_relevant_person(person_data: Dict[str, Any]) -> bool:
    """Check if person is relevant based on HISCO."""
    occupation = person_data.get("occupation", {})
    if not occupation: return False
    hisco_code = occupation.get("hisco_code_swedish") or occupation.get("hisco_code_english")
    return check_hisco_range(hisco_code, RELEVANT_HISCO_RANGES)

def extract_birth_decade(birth_date: Optional[str]) -> Optional[int]:
    """Extract birth decade from date string."""
    if not birth_date or not isinstance(birth_date, str): return None
    year: Optional[int] = None
    year_match = re.search(r'(?:^|\D)(\d{4})(?:$|\D)', birth_date) 
    if year_match:
        try: year = int(year_match.group(1))
        except ValueError: pass
    if year and 1700 < year < 2020: return (year // 10) * 10
    return None

def classify_education(education_entries: Optional[List[Dict[str, Any]]]) -> Dict[str, bool]:
    """Classify education into types."""
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
    """Identify overseas and US experience."""
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

def parse_json_string(json_string: Optional[str], default_value: Any = []) -> Any:
    """Safely parse a JSON string, returning a default value on error."""
    if pd.isna(json_string) or not isinstance(json_string, str): return default_value
    try:
        if not json_string.strip(): return default_value
        return json.loads(json_string)
    except json.JSONDecodeError: return default_value
    
# **** UPDATED HISCO Classification Function ****
def get_hisco_major_group_label(hisco_str: Optional[str]) -> str:
    """Determines HISCO major group based on 4/5 digit rule and maps to label."""
    if hisco_str is None or not isinstance(hisco_str, str): return "Unknown"
    cleaned_hisco = re.sub(r'\D', '', hisco_str) 
    major_group_num: Optional[int] = None

    if len(cleaned_hisco) == 4: major_group_num = 0
    elif len(cleaned_hisco) == 5:
        try: major_group_num = int(cleaned_hisco[0])
        except (ValueError, IndexError): return "Unknown"
    else: return "Unknown"

    # Map based on image provided
    if major_group_num in [0, 1]: return "Professional/Technical" 
    elif major_group_num == 2: return "Administrative/Managerial"
    elif major_group_num == 3: return "Clerical"
    elif major_group_num == 4: return "Sales"
    elif major_group_num == 5: return "Service"
    elif major_group_num == 6: return "Agricultural/Fishing"
    elif major_group_num in [7, 8, 9]: return "Production/Transport/Laborer"
    else: return "Unknown"

def standardize_education_data(education_list: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Standardizes institution names within a list of education entries."""
    standardized_list = []
    for entry in education_list:
        new_entry = entry.copy() 
        original_name = new_entry.get("institution")
        std_name = "Unknown/Unspecified" # Default
        if original_name and isinstance(original_name, str):
            normalized_original = original_name.strip()
            std_name = mapping.get(normalized_original, normalized_original) # Default to original if not mapped
        new_entry["institution_standardized"] = std_name
        standardized_list.append(new_entry)
    return standardized_list

# --- Data Extraction Function ---
def extract_persons_data() -> List[Dict[str, Any]]:
    """Extract key information, keeping education/career as lists."""
    persons_data_list: List[Dict[str, Any]] = []
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return persons_data_list
    all_files: List[str] = glob(os.path.join(INPUT_DIR, "*.json"))
    logging.info(f"Found {len(all_files)} potential JSON files.")
    files_processed: int = 0
    for filename in all_files:
        files_processed += 1
        if files_processed % 500 == 0: logging.info(f"Processed {files_processed}/{len(all_files)} files...")
        try:
            with open(filename, 'r', encoding='utf-8') as f: data = json.load(f)
            person = data.get('person', {})
            if not person or not is_relevant_person(person): continue 
            # Extract details...
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
                                   'hisco_code': (p.get('occupation') or {}).get('hisco_code_swedish') or (p.get('occupation') or {}).get('hisco_code_english')}
                         break
            education_entries: List[Dict] = data.get('education', []) or []
            education_classification: Dict[str, bool] = classify_education(education_entries)
            career_entries: List[Dict] = data.get('career', []) or []
            has_overseas, has_us, overseas_countries = get_overseas_experience(career_entries)
            board_count: int = len(data.get('board_memberships', []) or [])
            # Create record...
            person_record: Dict[str, Any] = {
                'person_id': os.path.basename(filename).replace('.json', ''),
                'person_name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                'birth_date': person.get('birth_date'), 'birth_decade': extract_birth_decade(person.get('birth_date')),
                'birth_location_name': birth_loc["name"], 'birth_location_lat': birth_loc["lat"], 'birth_location_lon': birth_loc["lon"], 
                'current_location_name': current_loc["name"], 'current_location_lat': current_loc["lat"], 'current_location_lon': current_loc["lon"],
                'person_occupation': occupation.get('occupation'), 'person_hisco_swe': occupation.get('hisco_code_swedish'), 'person_hisco_eng': occupation.get('hisco_code_english'),
                'father_name': father.get('name'), 'father_occupation': father.get('occupation'), 'father_hisco': father.get('hisco_code'),
                'edu_technical': education_classification['technical'], 'edu_business': education_classification['business'], 'edu_other_higher': education_classification['other_higher'],
                'career_has_overseas': has_overseas, 'career_has_us': has_us, 'career_overseas_countries': ",".join(overseas_countries), 
                'board_membership_count': board_count,
                '_education_raw': education_entries, # Keep raw list
                '_career_raw': career_entries, # Keep raw list
             }
            persons_data_list.append(person_record)
        except Exception as e: logging.error(f"Error processing file {os.path.basename(filename)}: {e}", exc_info=False)
    logging.info(f"Completed initial data extraction. Found {len(persons_data_list)} relevant persons.")
    return persons_data_list

# --- Function to Map Unique Locations ---
def map_unique_locations(locations_df: pd.DataFrame, parishes_gdf: gpd.GeoDataFrame) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Maps unique lat/lon pairs to parish information."""
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
            wl_status: Optional[bool] = None 
            if pd.notna(wl_status_raw):
                 try: wl_status = bool(int(wl_status_raw))
                 except (ValueError, TypeError): pass 
            location_map[(lat, lon)] = { 'parish_code': row.get(SHP_PARISH_CODE_COL), 
                                         'parish_name': row.get(SHP_PARISH_NAME_COL), 
                                         'is_western_line': wl_status }
        logging.info(f"Finished mapping unique locations.")
    except Exception as e: logging.error(f"Error during unique location mapping: {e}", exc_info=True)
    return location_map

# --- Function to Calculate Network Variables ---
def calculate_education_networks(df: pd.DataFrame) -> pd.DataFrame:
    """ Calculates educational network variables."""
    logging.info("Building institution-student lookup for network calculation...")
    institution_lookup = defaultdict(list)
    required_cols = ['education_standardized_parsed', 'person_id', 'birth_parish_is_western_line']
    if not all(col in df.columns for col in required_cols):
         logging.error("Input DataFrame missing required columns for network calculation.")
         df['edu_network_size'] = 0; df['edu_network_wl_birth_prop'] = np.nan
         return df
    for idx, row in df.iterrows():
        person_id = row['person_id']; born_wl = row['birth_parish_is_western_line']
        education_list = row.get('education_standardized_parsed', [])
        if not isinstance(education_list, list): continue
        for edu_entry in education_list:
             inst_std = edu_entry.get("institution_standardized"); year_str = edu_entry.get("year")
             year = pd.to_numeric(year_str, errors='coerce')
             if pd.notna(inst_std) and pd.notna(year) and isinstance(inst_std, str):
                 institution_lookup[inst_std].append((person_id, int(year), born_wl))
    logging.info(f"Institution lookup built with {len(institution_lookup)} institutions.")
    network_sizes, network_wl_props = [], []
    logging.info("Calculating network variables for each person...")
    for idx, row in df.iterrows():
        person_id = row['person_id']; max_network_size = 0
        valid_wl_proportions = []; primary_edu_entries = []
        education_list = row.get('education_standardized_parsed', [])
        if not isinstance(education_list, list): 
            network_sizes.append(0); network_wl_props.append(np.nan); continue
        latest_grad_year = -1; temp_primary_edu = []
        for edu_entry in education_list:
            inst_std = edu_entry.get("institution_standardized"); year_str = edu_entry.get("year")
            year = pd.to_numeric(year_str, errors='coerce')
            if pd.notna(inst_std) and pd.notna(year) and isinstance(inst_std, str):
                year = int(year); temp_primary_edu.append({'inst': inst_std, 'year': year})
                if year > latest_grad_year: latest_grad_year = year
        if latest_grad_year > 0: primary_edu_entries = [edu for edu in temp_primary_edu if edu['year'] == latest_grad_year]
        if not primary_edu_entries: network_sizes.append(0); network_wl_props.append(np.nan); continue
        person_cohort_sizes = []; person_cohort_wl_props = []
        for primary_edu in primary_edu_entries:
             inst_std = primary_edu['inst']; year = primary_edu['year']
             cohort = []
             if inst_std in institution_lookup:
                 min_year = year - COHORT_YEAR_WINDOW; max_year = year + COHORT_YEAR_WINDOW
                 for peer_id, peer_year, peer_born_wl in institution_lookup[inst_std]:
                     if min_year <= peer_year <= max_year and peer_id != person_id:
                         cohort.append({'id': peer_id, 'born_wl': peer_born_wl})
             current_network_size = len(cohort)
             person_cohort_sizes.append(current_network_size)
             if current_network_size > 0:
                 wl_peers = sum(1 for peer in cohort if peer['born_wl'] is True) 
                 wl_proportion = wl_peers / current_network_size
                 person_cohort_wl_props.append(wl_proportion)
        network_sizes.append(max(person_cohort_sizes) if person_cohort_sizes else 0)
        network_wl_props.append(np.mean(person_cohort_wl_props) if person_cohort_wl_props else np.nan)
    df['edu_network_size'] = network_sizes
    df['edu_network_wl_birth_prop'] = network_wl_props
    logging.info("Finished calculating network variables.")
    return df

# --- Main Execution ---
def main() -> int:
    """Main execution function."""
    logging.info(f"Starting data extraction, mapping, and standardization...")
    # --- 1. Load Institution Mapping ---
    inst_mapping: Dict[str, str] = {}
    if not os.path.exists(INSTITUTION_MAPPING_PATH): logging.warning(f"Institution mapping file not found: {INSTITUTION_MAPPING_PATH}.")
    else:
        try:
            with open(INSTITUTION_MAPPING_PATH, 'r', encoding='utf-8') as f: inst_mapping = json.load(f)
            logging.info(f"Loaded institution mapping ({len(inst_mapping)} entries).")
        except Exception as e: logging.error(f"Failed to load institution mapping: {e}.")
            
    # --- 2. Extract Raw Data ---
    persons_data_list: List[Dict[str, Any]] = extract_persons_data()
    if not persons_data_list: return 1
        
    # --- 3. Load Parish Shapefile ---
    if not os.path.exists(PARISH_SHAPEFILE_PATH): logging.error(f"Parish shapefile not found: {PARISH_SHAPEFILE_PATH}."); return 1
    try:
        parishes_gdf = gpd.read_file(PARISH_SHAPEFILE_PATH)
        if not parishes_gdf.crs: parishes_gdf.set_crs("EPSG:3006", inplace=True)
        logging.info(f"Loaded parishes shapefile (CRS: {parishes_gdf.crs})")
        required_shp_cols: List[str] = [SHP_PARISH_CODE_COL, SHP_PARISH_NAME_COL, SHP_WESTERN_LINE_COL, SHP_GEOMETRY_COL]
        if not all(col in parishes_gdf.columns for col in required_shp_cols):
             logging.error(f"Parish shapefile missing required columns."); return 1
    except Exception as e: logging.error(f"Failed to load parish shapefile: {e}"); return 1

    # --- 4. Consolidate and Map Unique Locations ---
    unique_locations: Set[Tuple[float, float]] = set()
    for person in persons_data_list:
        for loc_type in ['birth_location', 'current_location']:
            lat, lon = person.get(f"{loc_type}_lat"), person.get(f"{loc_type}_lon")
            if pd.notna(lat) and pd.notna(lon): unique_locations.add((lat, lon))
        for job in person.get('_career_raw', []): # Uses raw list here
            loc = get_location_details(job.get('location'))
            if pd.notna(loc['lat']) and pd.notna(loc['lon']): unique_locations.add((loc['lat'], loc['lon']))
        for edu in person.get('_education_raw', []): # Uses raw list here
            loc = get_location_details(edu.get('location'))
            if pd.notna(loc['lat']) and pd.notna(loc['lon']): unique_locations.add((loc['lat'], loc['lon']))
    if not unique_locations: location_parish_lookup = {}
    else:
         unique_locations_df = pd.DataFrame(list(unique_locations), columns=['lat', 'lon'])
         location_parish_lookup = map_unique_locations(unique_locations_df, parishes_gdf)

    # --- 5. Process Each Person Dictionary ---
    logging.info("Assigning mapped info, standardizing education, calculating flags...")
    processed_persons_list: List[Dict[str, Any]] = []
    for person in persons_data_list: # person is a dict from the initial extraction
        # Assign spatial info
        birth_coords = (person.get('birth_location_lat'), person.get('birth_location_lon'))
        birth_parish_info = location_parish_lookup.get(birth_coords, {}) 
        person['birth_parish_code'] = birth_parish_info.get('parish_code')
        person['birth_parish_name'] = birth_parish_info.get('parish_name')
        person['birth_parish_is_western_line'] = birth_parish_info.get('is_western_line') 
        current_coords = (person.get('current_location_lat'), person.get('current_location_lon'))
        current_parish_info = location_parish_lookup.get(current_coords, {})
        person['currently_lives_in_wl'] = current_parish_info.get('is_western_line') 
        
        # Calculate career flags
        worked_wl_before = False; worked_wl_after = False
        raw_career_list = person.get('_career_raw', []) # Use raw list
        if isinstance(raw_career_list, list):
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
        person['worked_wl_before_1930'] = worked_wl_before
        person['worked_wl_after_1930'] = worked_wl_after
        
        # Standardize Education & keep raw as JSON string
        raw_edu_list = person.get('_education_raw', []) # Use raw list
        if isinstance(raw_edu_list, list):
            standardized_edu_list = standardize_education_data(raw_edu_list, inst_mapping) if inst_mapping else raw_edu_list
            person['education_standardized'] = json.dumps(standardized_edu_list, ensure_ascii=False)
            person['_education_raw'] = json.dumps(raw_edu_list, ensure_ascii=False) # Store original raw as JSON string
        else: # Handle case where raw data wasn't a list
             person['education_standardized'] = '[]'
             person['_education_raw'] = '[]'

        # Keep raw career as JSON string
        raw_career_list_for_json = person.get('_career_raw', [])
        person['_career_raw'] = json.dumps(raw_career_list_for_json if isinstance(raw_career_list_for_json, list) else [], ensure_ascii=False)
        
        processed_persons_list.append(person) # Add processed dict to new list
        
    # --- 6. Create DataFrame and Add Final Calculated Columns ---
    final_df = pd.DataFrame(processed_persons_list)

    # Create Father HISCO Major Group Label
    logging.info("Creating father_hisco_major_group_label...")
    final_df['father_hisco_major_group_label'] = final_df['father_hisco'].apply(get_hisco_major_group_label)
    final_df['father_hisco_major_group_label'] = pd.Categorical(final_df['father_hisco_major_group_label'])
    logging.info(f"Father HISCO Major Group distribution:\n{final_df['father_hisco_major_group_label'].value_counts(dropna=False)}")
    
    # Create Studied At Dummies
    logging.info("Creating studied_at_* dummies...")
    kth_aliases = ["kungliga tekniska högskolan", "kth", "tekniska högskolan"] 
    chalmers_aliases = ["chalmers tekniska högskola", "chalmers", "cth", "chalmers tekniska institut", "cti", "chalmers tekniska läroanstalt"]
    hhs_aliases = ["handelshögskolan i stockholm", "handelshögskolan stockholm", "hhs", "handelshögskolan, stockholm"]
    foreign_strings = ["foreign study", "foreign university"] 
    
    def check_study_location_from_json(edu_json_str, targets):
        edu_list = parse_json_string(edu_json_str, [])
        if not isinstance(edu_list, list): return 0
        for entry in edu_list:
            # Use the standardized name for checking
            inst_std = entry.get("institution_standardized", "").lower() 
            if any(target in inst_std for target in targets): return 1
        return 0

    final_df['studied_kth'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, kth_aliases))
    final_df['studied_chalmers'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, chalmers_aliases))
    final_df['studied_hhs'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, hhs_aliases))
    final_df['studied_foreign'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, foreign_strings))
    logging.info(f"Studied KTH: {final_df['studied_kth'].sum()}, Chalmers: {final_df['studied_chalmers'].sum()}, HHS: {final_df['studied_hhs'].sum()}, Foreign: {final_df['studied_foreign'].sum()}")

    # Create Birth Cohort Category
    logging.info("Creating birth_cohort category...")
    bins = [0, 1880, 1900, 1920, np.inf]; labels = ['<1880', '1880-1899', '1900-1919', '1920+']
    final_df['birth_cohort'] = pd.cut(final_df['birth_decade'], bins=bins, labels=labels, right=False)
    logging.info(f"Birth cohort distribution:\n{final_df['birth_cohort'].value_counts(dropna=False)}")
    
    # Calculate Network Variables (needs parsed education data)
    final_df['education_standardized_parsed'] = final_df['education_standardized'].apply(parse_json_string)
    final_df = calculate_education_networks(final_df)
    final_df = final_df.drop(columns=['education_standardized_parsed'], errors='ignore') # Drop helper column

    # --- 7. Prepare Data for Regression ---
    logging.info("Final cleaning and preparation for regression models...")
    
    # Convert boolean columns to 0/1
    bool_cols_to_int = ['edu_technical', 'edu_business', 'edu_other_higher', 
                        'career_has_overseas', 'career_has_us', 
                        'worked_wl_before_1930', 'worked_wl_after_1930', # Already boolean but cast anyway
                        'studied_kth', 'studied_chalmers', 'studied_hhs', 'studied_foreign'] # Dummies
    for col in bool_cols_to_int:
        if col in final_df.columns: final_df[col] = final_df[col].astype(bool).astype(int)

    # Prepare Dependent Variable 1: Birth WL (1/0, drop unknowns)
    final_df['dep_var_birth_wl'] = final_df['birth_parish_is_western_line'].map({True: 1, False: 0}) # Maps True->1, False->0, None->NaN
    df_model1 = final_df.dropna(subset=['dep_var_birth_wl']).copy() # Create copy for model 1
    df_model1['dep_var_birth_wl'] = df_model1['dep_var_birth_wl'].astype(int)
    
    # Prepare Dependent Variable 2: Work WL Pre-1930 (already 0/1)
    final_df['dep_var_work_wl'] = final_df['worked_wl_before_1930'] 
    df_model2 = final_df.copy() # Create copy for model 2 (no rows dropped yet based on this DV)

    # Define independent variables list (ensure names match DataFrame columns)
    independent_vars = [
        'C(father_hisco_major_group_label)', 'C(birth_cohort)', 
        'edu_technical', 'edu_business', 'edu_other_higher', 
        'career_has_overseas', 'career_has_us', 'board_membership_count', 
        'studied_kth', 'studied_chalmers', 'studied_hhs', 'studied_foreign',
        'edu_network_size', 'edu_network_wl_birth_prop' 
    ]
    
    # Handle remaining NaNs in independent variables (Example: Drop rows)
    cols_to_check_na = [col for col in independent_vars if not col.startswith('C(')] # Check non-categorical
    # Add categorical columns explicitly if they might have NaNs *before* dummy creation (less likely here)
    cols_to_check_na.extend(['father_hisco_major_group_label', 'birth_cohort']) 
    
    initial_rows1 = len(df_model1)
    initial_rows2 = len(df_model2)
    df_model1 = df_model1.dropna(subset=cols_to_check_na)
    df_model2 = df_model2.dropna(subset=cols_to_check_na)
    logging.info(f"Rows for Model 1 after NA drop in predictors: {len(df_model1)} (out of {initial_rows1})")
    logging.info(f"Rows for Model 2 after NA drop in predictors: {len(df_model2)} (out of {initial_rows2})")


    # --- 8. Define and Run Probit Models ---
    formula_rhs = " + ".join(independent_vars)

    # --- Model 1: Predicting Birth in WL ---
    if not df_model1.empty:
        logging.info("\n--- Running Probit Model 1: Predicting Birth in WL Parish ---")
        formula1 = f"dep_var_birth_wl ~ {formula_rhs}"
        logging.info(f"Formula: {formula1}")
        try:
            probit_model1 = smf.probit(formula=formula1, data=df_model1)
            results1 = probit_model1.fit(maxiter=100) # Increased maxiter
            print("\n--- Probit Model 1 Summary ---")
            print(results1.summary())
        except Exception as e:
            logging.error(f"Error fitting Probit Model 1: {e}", exc_info=True)
    else:
        logging.warning("Skipping Probit Model 1: No data available after cleaning.")

    # --- Model 2: Predicting Work in WL before 1930 ---
    if not df_model2.empty:
        logging.info("\n--- Running Probit Model 2: Predicting Work in WL before 1930 ---")
        formula2 = f"dep_var_work_wl ~ {formula_rhs}"
        logging.info(f"Formula: {formula2}")
        try:
            probit_model2 = smf.probit(formula=formula2, data=df_model2)
            results2 = probit_model2.fit(maxiter=100) # Increased maxiter
            print("\n--- Probit Model 2 Summary ---")
            print(results2.summary())
        except Exception as e:
            logging.error(f"Error fitting Probit Model 2: {e}", exc_info=True)
    else:
        logging.warning("Skipping Probit Model 2: No data available after cleaning.")

    # --- 9. Save Final Processed DataFrame ---
    try:
        # Save the full dataframe *before* filtering/dropping for models
        final_df.to_csv(FINAL_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig') 
        logging.info(f"Saved final processed data ({len(final_df)} rows) to: {FINAL_CSV_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to save final CSV: {e}")
        return 1

    logging.info("Processing finished successfully.")
    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    # Check for dependencies
    try: import geopandas; logging.info(f"Geopandas version: {geopandas.__version__}")
    except ImportError: logging.error("Required module 'geopandas' not found."); exit(1) 
    try: import statsmodels; logging.info(f"Statsmodels version: {statsmodels.__version__}")
    except ImportError: logging.error("Required module 'statsmodels' not found."); exit(1) 
        
    exit_code = main()
    exit(exit_code)