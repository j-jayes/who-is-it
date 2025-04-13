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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_DIR: str = "data/enriched_biographies"
OUTPUT_DIR: str = "data/analysis"
PARISH_SHAPEFILE_PATH: str = "data/parishes/parish_map_1920.shp"
INSTITUTION_MAPPING_PATH: str = os.path.join(OUTPUT_DIR, "institution_mapping.json") 
FINAL_CSV_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_final_for_regression.csv') 

os.makedirs(OUTPUT_DIR, exist_ok=True)

ENGINEER_HISCO_RANGES: List[Tuple[int, int]] = [(1000, 3999)] 
DIRECTOR_HISCO_RANGES: List[Tuple[int, int]] = [(21000, 21900)] 
RELEVANT_HISCO_RANGES: List[Tuple[int, int]] = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES
CAREER_YEAR_THRESHOLD: int = 1930
SHP_PARISH_CODE_COL: str = 'prsh_cd'; SHP_PARISH_NAME_COL: str = 'prsh_nm'; SHP_WESTERN_LINE_COL: str = 'wstrn_l'; SHP_GEOMETRY_COL: str = 'geometry'
COHORT_YEAR_WINDOW: int = 4 


# --- Helper Functions (Assume they are defined correctly as in previous versions) ---
def safe_float_to_int(value: Any) -> Optional[int]:
    if value is None: return None
    try: return int(float(value))
    except (ValueError, TypeError): return None

def check_hisco_range(code_str: Optional[str], ranges: List[Tuple[int, int]]) -> bool:
    code_value = safe_float_to_int(code_str)
    if code_value is None: return False
    for min_val, max_val in ranges:
        if min_val <= code_value <= max_val: return True
    return False

def is_relevant_person(person_data: Dict[str, Any]) -> bool:
    occupation = person_data.get("occupation", {})
    if not occupation: return False
    hisco_code = occupation.get("hisco_code_swedish") or occupation.get("hisco_code_english")
    return check_hisco_range(hisco_code, RELEVANT_HISCO_RANGES)

def extract_birth_decade(birth_date: Optional[str]) -> Optional[int]:
    if not birth_date or not isinstance(birth_date, str): return None
    year: Optional[int] = None
    year_match = re.search(r'(?:^|\D)(\d{4})(?:$|\D)', birth_date) 
    if year_match:
        try: year = int(year_match.group(1))
        except ValueError: pass
    if year and 1700 < year < 2020: return (year // 10) * 10
    return None

def classify_education(education_entries: Optional[List[Dict[str, Any]]]) -> Dict[str, bool]:
    TECH_KW: List[str] = ['tekniska', 'chalmers', 'kth', 'tekn', 'ingenjör', 'teknisk', 'teknolog', 'polytekn', 'engineering', 'technical']
    BIZ_KW: List[str] = ['handels', 'ekonomi', 'handelshögskola', 'business', 'commerce', 'ekonomisk', 'handelsinstitut', 'handelsgymnasium', 'ekonom']
    HIGHER_EDU_LEVELS: List[str] = ["Master's", "PhD", "Bachelor's", "Licentiate"]
    HIGHER_EDU_KW: List[str] = ['högskola', 'universitet', 'university', 'college']
    if not education_entries: return {"technical": False, "business": False, "other_higher": False}
    result: Dict[str, bool] = {"technical": False, "business": False, "other_higher": False}; has_higher_edu: bool = False
    for entry in education_entries:
        institution: str = str(entry.get("institution", "")).lower(); degree: str = str(entry.get("degree", "")).lower(); full_text: str = institution + " " + degree
        is_higher: bool = entry.get("degree_level") in HIGHER_EDU_LEVELS or any(keyword in full_text for keyword in HIGHER_EDU_KW)
        if is_higher: has_higher_edu = True
        if any(keyword in full_text for keyword in TECH_KW): result["technical"] = True
        elif any(keyword in full_text for keyword in BIZ_KW): result["business"] = True
        elif is_higher: result["other_higher"] = True
    if not has_higher_edu: result["other_higher"] = False
    return result

def get_location_details(location_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not location_data or not isinstance(location_data, dict): return {"name": None, "lat": None, "lon": None}
    lat = pd.to_numeric(location_data.get("latitude"), errors='coerce'); lon = pd.to_numeric(location_data.get("longitude"), errors='coerce')
    lat_valid = pd.notna(lat) and -90 <= lat <= 90; lon_valid = pd.notna(lon) and -180 <= lon <= 180
    return {"name": location_data.get("name"), "lat": lat if lat_valid else None, "lon": lon if lon_valid else None}

def get_overseas_experience(career_entries: Optional[List[Dict[str, Any]]]) -> Tuple[bool, bool, List[str]]:
    has_overseas, has_us, countries = False, False, set()
    if not career_entries: return has_overseas, has_us, []
    for entry in career_entries:
        country_code = entry.get("country_code")
        if country_code and isinstance(country_code, str) and country_code.upper() != "SWE":
            has_overseas = True; cc_upper = country_code.upper(); countries.add(cc_upper)
            if cc_upper == "USA": has_us = True
    return has_overseas, has_us, sorted(list(countries))

def parse_json_string(json_string: Optional[str], default_value: Any = []) -> Any:
    if pd.isna(json_string) or not isinstance(json_string, str): return default_value
    try:
        if not json_string.strip(): return default_value
        return json.loads(json_string)
    except json.JSONDecodeError: return default_value

# Updated HISCO Classification Function
def get_hisco_major_group_label(hisco_str: Optional[str]) -> str:
    if hisco_str is None or not isinstance(hisco_str, str): return "Unknown"
    cleaned_hisco = re.sub(r'\D', '', hisco_str); major_group_num: Optional[int] = None
    if len(cleaned_hisco) == 4: major_group_num = 0
    elif len(cleaned_hisco) == 5:
        try: major_group_num = int(cleaned_hisco[0])
        except (ValueError, IndexError): return "Unknown"
    else: return "Unknown"
    if major_group_num in [0, 1]: return "Professional/Technical" 
    elif major_group_num == 2: return "Administrative/Managerial"
    elif major_group_num == 3: return "Clerical"
    elif major_group_num == 4: return "Sales"
    elif major_group_num == 5: return "Service"
    elif major_group_num == 6: return "Agricultural/Fishing"
    elif major_group_num in [7, 8, 9]: return "Production/Transport/Laborer"
    else: return "Unknown" 

# Institution Standardization Function
def standardize_education_data(education_list: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    standardized_list = []
    for entry in education_list:
        new_entry = entry.copy(); original_name = new_entry.get("institution"); std_name = "Unknown/Unspecified"
        if original_name and isinstance(original_name, str):
            normalized_original = original_name.strip(); std_name = mapping.get(normalized_original, normalized_original) 
        new_entry["institution_standardized"] = std_name
        standardized_list.append(new_entry)
    return standardized_list

# Data Extraction Function
def extract_persons_data() -> List[Dict[str, Any]]:
    persons_data_list: List[Dict[str, Any]] = []
    if not os.path.isdir(INPUT_DIR): logging.error(f"Input directory not found: {INPUT_DIR}"); return persons_data_list
    all_files: List[str] = glob(os.path.join(INPUT_DIR, "*.json")); logging.info(f"Found {len(all_files)} potential JSON files.")
    files_processed: int = 0
    for filename in all_files:
        files_processed += 1
        if files_processed % 500 == 0: logging.info(f"Processed {files_processed}/{len(all_files)} files...")
        try:
            with open(filename, 'r', encoding='utf-8') as f: data = json.load(f)
            person = data.get('person', {}); 
            if not person or not is_relevant_person(person): continue 
            birth_loc = get_location_details(person.get('birth_place')); current_loc = get_location_details(data.get('current_location'))
            occupation = person.get('occupation', {}) or {}; father: Dict[str, Any] = {}
            parents = person.get('parents', [])
            if parents:
                 for p in parents:
                     if p and p.get('gender') == 'Male':
                         father = {'name': f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                                   'occupation': (p.get('occupation') or {}).get('occupation'),
                                   'hisco_code': (p.get('occupation') or {}).get('hisco_code_swedish') or (p.get('occupation') or {}).get('hisco_code_english') }
                         break
            education_entries: List[Dict] = data.get('education', []) or []; education_classification: Dict[str, bool] = classify_education(education_entries)
            career_entries: List[Dict] = data.get('career', []) or []; has_overseas, has_us, overseas_countries = get_overseas_experience(career_entries)
            board_count: int = len(data.get('board_memberships', []) or [])
            person_record: Dict[str, Any] = {
                'person_id': os.path.basename(filename).replace('.json', ''), 'person_name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                'birth_date': person.get('birth_date'), 'birth_decade': extract_birth_decade(person.get('birth_date')),
                'birth_location_name': birth_loc["name"], 'birth_location_lat': birth_loc["lat"], 'birth_location_lon': birth_loc["lon"], 
                'current_location_name': current_loc["name"], 'current_location_lat': current_loc["lat"], 'current_location_lon': current_loc["lon"],
                'person_occupation': occupation.get('occupation'), 'person_hisco_swe': occupation.get('hisco_code_swedish'), 'person_hisco_eng': occupation.get('hisco_code_english'),
                'father_name': father.get('name'), 'father_occupation': father.get('occupation'), 'father_hisco': father.get('hisco_code'),
                'edu_technical': education_classification['technical'], 'edu_business': education_classification['business'], 'edu_other_higher': education_classification['other_higher'],
                'career_has_overseas': has_overseas, 'career_has_us': has_us, 'career_overseas_countries': ",".join(overseas_countries), 'board_membership_count': board_count,
                '_education_raw': education_entries, '_career_raw': career_entries,
             }
            persons_data_list.append(person_record)
        except Exception as e: logging.error(f"Error processing file {os.path.basename(filename)}: {e}", exc_info=False)
    logging.info(f"Completed initial data extraction. Found {len(persons_data_list)} relevant persons.")
    return persons_data_list

# Function to Map Unique Locations
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
        # Use inner join initially to diagnose - points outside polygons will be dropped
        # Change back to 'left' if needed, but 'inner' helps find alignment issues
        joined_gdf = gpd.sjoin(locations_gdf, parishes_simplified, how='inner', predicate='within') 
        logging.info(f"Spatial join mapped {len(joined_gdf)} unique locations.")
        
        for original_idx, row in joined_gdf.iterrows():
            lat, lon = locations_df.loc[original_idx, 'lat'], locations_df.loc[original_idx, 'lon']
            wl_status_raw = row.get(SHP_WESTERN_LINE_COL); wl_status: Optional[bool] = None 
            if pd.notna(wl_status_raw):
                 try: wl_status = bool(int(wl_status_raw))
                 except (ValueError, TypeError): pass 
            location_map[(lat, lon)] = { 'parish_code': row.get(SHP_PARISH_CODE_COL), 'parish_name': row.get(SHP_PARISH_NAME_COL), 'is_western_line': wl_status }
        logging.info(f"Finished mapping unique locations. Created lookup for {len(location_map)} coordinates.")
    except Exception as e: logging.error(f"Error during unique location mapping: {e}", exc_info=True)
    return location_map

# Function to Calculate Network Variables
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
        person_id = row['person_id']; born_wl = row['birth_parish_is_western_line'] # This is True/False/None
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
        person_id = row['person_id']; max_network_size = 0; valid_wl_proportions = []
        education_list = row.get('education_standardized_parsed', [])
        if not isinstance(education_list, list): network_sizes.append(0); network_wl_props.append(np.nan); continue
        latest_grad_year = -1; temp_primary_edu = []
        for edu_entry in education_list: # Find latest graduation year
            inst_std = edu_entry.get("institution_standardized"); year_str = edu_entry.get("year")
            year = pd.to_numeric(year_str, errors='coerce')
            if pd.notna(inst_std) and pd.notna(year) and isinstance(inst_std, str):
                year = int(year); temp_primary_edu.append({'inst': inst_std, 'year': year})
                if year > latest_grad_year: latest_grad_year = year
        primary_edu_entries = [edu for edu in temp_primary_edu if edu['year'] == latest_grad_year] if latest_grad_year > 0 else []
        if not primary_edu_entries: network_sizes.append(0); network_wl_props.append(np.nan); continue
        person_cohort_sizes = []; person_cohort_wl_props = []
        for primary_edu in primary_edu_entries: # Calculate for latest year entries
             inst_std = primary_edu['inst']; year = primary_edu['year']; cohort = []
             if inst_std in institution_lookup:
                 min_year = year - COHORT_YEAR_WINDOW; max_year = year + COHORT_YEAR_WINDOW
                 for peer_id, peer_year, peer_born_wl in institution_lookup[inst_std]:
                     if min_year <= peer_year <= max_year and peer_id != person_id: cohort.append({'id': peer_id, 'born_wl': peer_born_wl})
             current_network_size = len(cohort)
             person_cohort_sizes.append(current_network_size)
             if current_network_size > 0:
                 # Count only non-None born_wl values for denominator
                 known_status_peers = [p for p in cohort if p['born_wl'] is not None]
                 wl_peers = sum(1 for peer in known_status_peers if peer['born_wl'] is True)
                 wl_proportion = wl_peers / len(known_status_peers) if known_status_peers else 0 # Avoid division by zero
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
        parishes_gdf = gpd.read_file(PARISH_SHAPEFILE_PATH); 
        if not parishes_gdf.crs: parishes_gdf.set_crs("EPSG:3006", inplace=True)
        logging.info(f"Loaded parishes shapefile (CRS: {parishes_gdf.crs})")
        required_shp_cols: List[str] = [SHP_PARISH_CODE_COL, SHP_PARISH_NAME_COL, SHP_WESTERN_LINE_COL, SHP_GEOMETRY_COL]
        if not all(col in parishes_gdf.columns for col in required_shp_cols): logging.error(f"Parish shapefile missing required columns."); return 1
    except Exception as e: logging.error(f"Failed to load parish shapefile: {e}"); return 1

    # --- 4. Consolidate and Map Unique Locations ---
    unique_locations: Set[Tuple[float, float]] = set()
    for person in persons_data_list:
        for loc_type in ['birth_location', 'current_location']: lat, lon = person.get(f"{loc_type}_lat"), person.get(f"{loc_type}_lon");_ = (pd.notna(lat) and pd.notna(lon)) and unique_locations.add((lat, lon))
        for job in person.get('_career_raw', []): loc = get_location_details(job.get('location')); _ = (pd.notna(loc['lat']) and pd.notna(loc['lon'])) and unique_locations.add((loc['lat'], loc['lon']))
        for edu in person.get('_education_raw', []): loc = get_location_details(edu.get('location')); _ = (pd.notna(loc['lat']) and pd.notna(loc['lon'])) and unique_locations.add((loc['lat'], loc['lon']))
    location_parish_lookup = map_unique_locations(pd.DataFrame(list(unique_locations), columns=['lat', 'lon']), parishes_gdf) if unique_locations else {}

    # --- 5. Process Each Person Dictionary ---
    logging.info("Assigning mapped info, standardizing education, calculating flags...")
    processed_persons_list: List[Dict[str, Any]] = []
    for person in persons_data_list: 
        birth_coords = (person.get('birth_location_lat'), person.get('birth_location_lon')); birth_parish_info = location_parish_lookup.get(birth_coords, {}) 
        person['birth_parish_code'] = birth_parish_info.get('parish_code'); person['birth_parish_name'] = birth_parish_info.get('parish_name')
        person['birth_parish_is_western_line'] = birth_parish_info.get('is_western_line') 
        current_coords = (person.get('current_location_lat'), person.get('current_location_lon')); current_parish_info = location_parish_lookup.get(current_coords, {})
        person['currently_lives_in_wl'] = current_parish_info.get('is_western_line') 
        worked_wl_before = False; worked_wl_after = False
        raw_career_list = person.get('_career_raw', [])
        if isinstance(raw_career_list, list):
            for job in raw_career_list:
                loc = get_location_details(job.get('location')); job_coords = (loc['lat'], loc['lon']); job_parish_info = location_parish_lookup.get(job_coords)
                if job_parish_info and job_parish_info.get('is_western_line') is True:
                    start_year = pd.to_numeric(job.get('start_year'), errors='coerce'); end_year = pd.to_numeric(job.get('end_year'), errors='coerce')
                    if pd.notna(start_year):
                        if start_year < CAREER_YEAR_THRESHOLD: worked_wl_before = True
                        if start_year >= CAREER_YEAR_THRESHOLD or (pd.notna(end_year) and end_year >= CAREER_YEAR_THRESHOLD) or pd.isna(end_year): worked_wl_after = True
        person['worked_wl_before_1930'] = worked_wl_before; person['worked_wl_after_1930'] = worked_wl_after
        raw_edu_list = person.get('_education_raw', [])
        standardized_edu_list = standardize_education_data(raw_edu_list, inst_mapping) if inst_mapping and isinstance(raw_edu_list, list) else raw_edu_list
        person['education_standardized'] = json.dumps(standardized_edu_list, ensure_ascii=False)
        person['_education_raw'] = json.dumps(raw_edu_list if isinstance(raw_edu_list, list) else [], ensure_ascii=False)
        person['_career_raw'] = json.dumps(raw_career_list if isinstance(raw_career_list, list) else [], ensure_ascii=False)
        processed_persons_list.append(person)
        
    # --- 6. Create DataFrame and Add Final Calculated Columns ---
    final_df = pd.DataFrame(processed_persons_list)
    logging.info("Creating derived variables (father class, studied at, cohort)...")
    final_df['father_hisco_major_group_label'] = final_df['father_hisco'].apply(get_hisco_major_group_label).astype('category')
    kth_aliases = ["kungliga tekniska högskolan", "kth", "tekniska högskolan"]; chalmers_aliases = ["chalmers tekniska högskola", "chalmers", "cth", "chalmers tekniska institut", "cti", "chalmers tekniska läroanstalt"]
    hhs_aliases = ["handelshögskolan i stockholm", "handelshögskolan stockholm", "hhs", "handelshögskolan, stockholm"]; foreign_strings = ["foreign study", "foreign university"] 
    def check_study_location_from_json(edu_json_str, targets): edu_list = parse_json_string(edu_json_str, []); inst_std = ""; return next((1 for entry in edu_list if isinstance(entry, dict) and any(target in str(entry.get("institution_standardized", "")).lower() for target in targets)), 0) if isinstance(edu_list, list) else 0
    final_df['studied_kth'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, kth_aliases))
    final_df['studied_chalmers'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, chalmers_aliases))
    final_df['studied_hhs'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, hhs_aliases))
    final_df['studied_foreign'] = final_df['education_standardized'].apply(lambda x: check_study_location_from_json(x, foreign_strings))
    bins = [0, 1880, 1900, 1920, np.inf]; labels = ['<1880', '1880-1899', '1900-1919', '1920+']
    final_df['birth_cohort'] = pd.cut(final_df['birth_decade'], bins=bins, labels=labels, right=False).astype('category')
    
    # Calculate Network Variables 
    final_df['education_standardized_parsed'] = final_df['education_standardized'].apply(parse_json_string)
    final_df = calculate_education_networks(final_df)
    final_df = final_df.drop(columns=['education_standardized_parsed'], errors='ignore') 

    # --- 7. Prepare Data for Regression ---
    logging.info("Final cleaning and preparation for regression models...")
    bool_cols_to_int = ['edu_technical', 'edu_business', 'edu_other_higher', 'career_has_overseas', 'career_has_us', 'worked_wl_before_1930', 'worked_wl_after_1930', 'studied_kth', 'studied_chalmers', 'studied_hhs', 'studied_foreign']
    for col in bool_cols_to_int:
        if col in final_df.columns: final_df[col] = final_df[col].astype(bool).astype(int)
    final_df['dep_var_birth_wl'] = final_df['birth_parish_is_western_line'].map({True: 1, False: 0}); df_model1 = final_df.dropna(subset=['dep_var_birth_wl']).copy(); df_model1['dep_var_birth_wl'] = df_model1['dep_var_birth_wl'].astype(int)
    final_df['dep_var_work_wl'] = final_df['worked_wl_before_1930']; df_model2 = final_df.copy() 
    
    # Define independent variables & Handle remaining NaNs
    independent_vars = ['C(father_hisco_major_group_label)', 'C(birth_cohort)', 'edu_technical', 'edu_business', 'edu_other_higher', 'career_has_overseas', 'career_has_us', 'board_membership_count', 'studied_kth', 'studied_chalmers', 'studied_hhs', 'studied_foreign','edu_network_size', 'edu_network_wl_birth_prop' ]
    predictors_for_na_check = [col for col in independent_vars if not col.startswith('C(')] # Check continuous/dummy vars for NA
    predictors_for_na_check.extend(['father_hisco_major_group_label', 'birth_cohort']) # Also check categoricals before dummy creation by statsmodels
    
    # **** TROUBLESHOOTING: Impute network proportion with median ****
    median_net_prop = df_model1['edu_network_wl_birth_prop'].median() # Use median from model 1's data
    logging.info(f"Imputing NaN in 'edu_network_wl_birth_prop' with median: {median_net_prop:.4f}")
    df_model1['edu_network_wl_birth_prop'].fillna(median_net_prop, inplace=True)
    df_model2['edu_network_wl_birth_prop'].fillna(median_net_prop, inplace=True) # Impute in model 2 data as well

    # Now drop rows only if other essential predictors are missing (less likely now)
    # Check if 'father_hisco_major_group_label' or 'birth_cohort' have actual NaNs (not 'Unknown')
    essential_predictors = ['father_hisco_major_group_label', 'birth_cohort'] # Add others if critical
    initial_rows1 = len(df_model1); initial_rows2 = len(df_model2)
    df_model1 = df_model1.dropna(subset=essential_predictors) 
    df_model2 = df_model2.dropna(subset=essential_predictors)
    logging.info(f"Rows for Model 1 after essential NA drop: {len(df_model1)} (out of {initial_rows1})")
    logging.info(f"Rows for Model 2 after essential NA drop: {len(df_model2)} (out of {initial_rows2})")
    logging.info(f"Missing values count for edu_network_wl_birth_prop after imputation: Model 1={df_model1['edu_network_wl_birth_prop'].isna().sum()}, Model 2={df_model2['edu_network_wl_birth_prop'].isna().sum()}")

    # --- 8. Define and Run Probit Models ---
    formula_rhs = " + ".join(independent_vars)
    
    # --- Model 1 ---
    if not df_model1.empty:
        logging.info("\n--- Running Probit Model 1: Predicting Birth in WL Parish ---")
        formula1 = f"dep_var_birth_wl ~ {formula_rhs}"
        logging.info(f"Formula: {formula1}")
        # **** TROUBLESHOOTING: Check dependent variable distribution ****
        logging.info(f"Model 1 Dep Var Distribution:\n{df_model1['dep_var_birth_wl'].value_counts(normalize=True)}")
        if df_model1['dep_var_birth_wl'].nunique() < 2:
             logging.error("Model 1 dependent variable has only one outcome. Cannot fit Probit model.")
        else:
            try:
                # Try standard solver first, then others if needed
                probit_model1 = smf.probit(formula=formula1, data=df_model1)
                results1 = probit_model1.fit(maxiter=200) # Increased maxiter further
                print("\n--- Probit Model 1 Summary ---"); print(results1.summary())
                # Add check for nan results after fitting
                if np.isnan(results1.llf):
                     logging.warning("Model 1 resulted in NaN log-likelihood. Trying alternative solver 'bfgs'.")
                     results1 = probit_model1.fit(method='bfgs', maxiter=500)
                     print("\n--- Probit Model 1 Summary (BFGS Solver) ---"); print(results1.summary())
                     if np.isnan(results1.llf): logging.error("Model 1 still failed with BFGS solver. Check for separation or data issues.")
            except Exception as e: logging.error(f"Error fitting Probit Model 1: {e}", exc_info=True)
    else: logging.warning("Skipping Probit Model 1: No data.")

    # --- Model 2 ---
    # **** TROUBLESHOOTING: Simplify model if quasi-separation suspected ****
    formula_rhs_model2 = formula_rhs # Start with full formula
    # Example: Comment out father's class if it caused issues previously
    # independent_vars_m2 = [v for v in independent_vars if 'father_hisco' not in v]
    # formula_rhs_model2 = " + ".join(independent_vars_m2)
    # logging.warning("Running Model 2 with simplified formula (excluding father's HISCO group)")

    if not df_model2.empty:
        logging.info("\n--- Running Probit Model 2: Predicting Work in WL before 1930 ---")
        formula2 = f"dep_var_work_wl ~ {formula_rhs_model2}"
        logging.info(f"Formula: {formula2}")
        # **** TROUBLESHOOTING: Check dependent variable distribution ****
        logging.info(f"Model 2 Dep Var Distribution:\n{df_model2['dep_var_work_wl'].value_counts(normalize=True)}")
        if df_model2['dep_var_work_wl'].nunique() < 2:
            logging.error("Model 2 dependent variable has only one outcome. Cannot fit Probit model.")
        else:
            try:
                probit_model2 = smf.probit(formula=formula2, data=df_model2)
                results2 = probit_model2.fit(maxiter=200) # Increased maxiter
                print("\n--- Probit Model 2 Summary ---"); print(results2.summary())
                # Check convergence warning
                if not results2.mle_retvals['converged']:
                     logging.warning("Model 2 failed to converge. Results may be unreliable. Check for quasi-separation.")
                     # Optionally, try simplifying the model further or using a different solver as for Model 1.
            except Exception as e: logging.error(f"Error fitting Probit Model 2: {e}", exc_info=True)
    else: logging.warning("Skipping Probit Model 2: No data.")

    # --- 9. Save Final Processed DataFrame ---
    try:
        final_df.to_csv(FINAL_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig') 
        logging.info(f"Saved final processed data ({len(final_df)} rows) to: {FINAL_CSV_OUTPUT_PATH}")
    except Exception as e: logging.error(f"Failed to save final CSV: {e}"); return 1

    logging.info("Processing finished successfully.")
    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    try: import geopandas; logging.info(f"Geopandas version: {geopandas.__version__}")
    except ImportError: logging.error("Required module 'geopandas' not found."); exit(1) 
    try: import statsmodels; logging.info(f"Statsmodels version: {statsmodels.__version__}")
    except ImportError: logging.error("Required module 'statsmodels' not found."); exit(1) 
    main()