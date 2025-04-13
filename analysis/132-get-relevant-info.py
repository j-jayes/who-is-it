import os
import json
import pandas as pd
import geopandas as gpd # Added for spatial operations
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
# Ensure this path points to your actual shapefile
PARISH_SHAPEFILE_PATH = "data/parishes/parish_map_1920.shp" 

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
# HISCO code ranges (adjust as needed based on your data's format and desired scope)
ENGINEER_HISCO_RANGES = [(1000, 3999)] 
DIRECTOR_HISCO_RANGES = [(21000, 21900)] 
RELEVANT_HISCO_RANGES = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES

MAX_EARLY_CAREER_ENTRIES = 3 

# Educational institution classification keywords
TECHNICAL_KEYWORDS = [
    'tekniska', 'chalmers', 'kth', 'tekn', 'ingenjör', 'teknisk', 
    'teknolog', 'polytekn', 'engineering', 'technical'
]
BUSINESS_KEYWORDS = [
    'handels', 'ekonomi', 'handelshögskola', 'business', 'commerce', 
    'ekonomisk', 'handelsinstitut', 'handelsgymnasium', 'ekonom'
]

# --- Helper Functions (largely unchanged) ---

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
    # Fallback check (optional)
    # occ_title = str(occupation.get('occupation', '')).lower()
    # if any(keyword in occ_title for keyword in ['ingenjör', 'engineer', 'direktör', 'director', 'manager', 'chef']): return True
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
    return {
        "name": location_data.get("name"),
        "lat": location_data.get("latitude"),
        "lon": location_data.get("longitude"),
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



# --- Data Extraction Function (mostly unchanged inside, returns list) ---

def extract_persons_data() -> List[Dict]:
    """Extract key information into a list of dictionaries."""
    persons_data_list = []
    # (Keep the inner loop logic from the previous version here)
    # ... (loop through files, parse json, check if relevant person) ...
    # Inside the loop, when a relevant person is found:
    # ... (extract all details: person_name, birth_loc, father, education, career, etc.) ...
    # --- Create Record (without placeholder columns initially) ---
    # **IMPORTANT**: Ensure 'birth_location_lat' and 'birth_location_lon' are correctly extracted
    
    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}")
        return []

    all_files = glob(os.path.join(INPUT_DIR, "*.json"))
    logging.info(f"Found {len(all_files)} potential JSON files in {INPUT_DIR}")
    files_processed = 0
    persons_count = 0
    
    for filename in all_files:
        files_processed += 1
        if files_processed % 500 == 0: logging.info(f"Processed {files_processed}/{len(all_files)} files...")

        try:
            with open(filename, 'r', encoding='utf-8') as f: data = json.load(f)
            person = data.get('person', {})
            if not person: continue 

            if is_relevant_person(person):
                persons_count += 1
                person_name = f"{person.get('first_name', '')} {person.get('last_name', '')}".strip()
                birth_date = person.get('birth_date')
                birth_decade = extract_birth_decade(birth_date)
                birth_loc = get_location_details(person.get('birth_place'))
                occupation = person.get('occupation', {}) or {}
                father = {}
                parents = person.get('parents', [])
                if parents:
                    for p in parents:
                        if p and p.get('gender') == 'Male':
                            father = { 'name': f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                                       'occupation': (p.get('occupation') or {}).get('occupation'),
                                       'hisco_code': (p.get('occupation') or {}).get('hisco_code_swedish') or (p.get('occupation') or {}).get('hisco_code_english') }
                            break
                education_entries = data.get('education', []) or []
                education_classification = classify_education(education_entries)
                education_details = []
                for edu in education_entries:
                    edu_loc = get_location_details(edu.get('location'))
                    education_details.append({"institution": edu.get("institution"), "degree": edu.get("degree"),
                                              "degree_level": edu.get("degree_level"), "year": edu.get("year"),
                                              "location_name": edu_loc["name"], "location_lat": edu_loc["lat"], "location_lon": edu_loc["lon"]})
                career_entries = data.get('career', []) or []
                has_overseas, has_us, overseas_countries = get_overseas_experience(career_entries)
                early_career = []
                sorted_career = sorted([c for c in career_entries if c.get('start_year')], key=lambda x: x['start_year'])
                for i, job in enumerate(sorted_career):
                    if i >= MAX_EARLY_CAREER_ENTRIES: break
                    job_loc = get_location_details(job.get('location'))
                    early_career.append({ "position": job.get("position"), "organization": job.get("organization"),
                                          "start_year": job.get("start_year"), "end_year": job.get("end_year"),
                                          "location_name": job_loc["name"], "location_lat": job_loc["lat"], "location_lon": job_loc["lon"],
                                          "country_code": job.get("country_code")})
                board_memberships = data.get('board_memberships', []) or []
                board_count = len(board_memberships)

                person_record = {
                    'person_id': os.path.basename(filename).replace('.json', ''),
                    'person_name': person_name,
                    'birth_date': birth_date,
                    'birth_decade': birth_decade,
                    'birth_location_name': birth_loc["name"],
                    'birth_location_lat': birth_loc["lat"], # Crucial for mapping
                    'birth_location_lon': birth_loc["lon"], # Crucial for mapping
                    'person_occupation': occupation.get('occupation'),
                    'person_hisco_swe': occupation.get('hisco_code_swedish'),
                    'person_hisco_eng': occupation.get('hisco_code_english'),
                    'father_name': father.get('name'),
                    'father_occupation': father.get('occupation'),
                    'father_hisco': father.get('hisco_code'),
                    'edu_technical': education_classification['technical'],
                    'edu_business': education_classification['business'],
                    'edu_other_higher': education_classification['other_higher'],
                    'education_details': json.dumps(education_details), 
                    'career_has_overseas': has_overseas,
                    'career_has_us': has_us,
                    'career_overseas_countries': ",".join(overseas_countries), 
                    'early_career_details': json.dumps(early_career), 
                    'board_membership_count': board_count,
                 }
                persons_data_list.append(person_record)
        except json.JSONDecodeError as e: logging.error(f"JSON error in {os.path.basename(filename)}: {e}")
        except Exception as e: logging.error(f"Error processing {os.path.basename(filename)}: {e}", exc_info=False) # Keep log concise

    logging.info(f"Completed initial data extraction. Found {len(persons_data_list)} relevant persons.")
    return persons_data_list


# --- NEW Function for Spatial Mapping ---
def map_birthplace_to_parish(persons_df: pd.DataFrame, shp_path: str) -> pd.DataFrame:
    """
    Performs spatial join to map person birth locations to parishes.

    Args:
        persons_df: DataFrame with person data including birth lat/lon columns.
        shp_path: Path to the parish shapefile.

    Returns:
        DataFrame with added columns for birth parish info (code, name, western_line).
        Returns the original DataFrame if spatial operations fail.
    """
    logging.info("Starting spatial mapping for birth locations...")
    
    # --- 1. Load Shapefile ---
    try:
        parishes_gdf = gpd.read_file(shp_path)
        logging.info(f"Loaded {len(parishes_gdf)} parishes from shapefile.")
        parish_crs = parishes_gdf.crs
        logging.info(f"Parish map CRS: {parish_crs}")
        if not parish_crs: # Handle missing CRS
             logging.warning("Parish shapefile CRS is missing. Assuming EPSG:3006.")
             parish_crs = "EPSG:3006"
             parishes_gdf.set_crs(parish_crs, inplace=True)
        # Check for required 'western_line' column (adjust name if needed)
        if 'wstrn_l' not in parishes_gdf.columns: 
             logging.error(f"Shapefile missing required 'wstrn_l' column. Found: {list(parishes_gdf.columns)}")
             return persons_df # Return original df if crucial column missing
             
    except Exception as e:
        logging.error(f"Error reading shapefile {shp_path}: {e}")
        return persons_df # Return original df on error

    # --- 2. Prepare Points GeoDataFrame ---
    lat_col, lon_col = 'birth_location_lat', 'birth_location_lon'
    if not all(col in persons_df.columns for col in [lat_col, lon_col]):
        logging.error(f"DataFrame missing required coordinate columns: {lat_col}, {lon_col}")
        return persons_df

    # Filter for valid coordinates and create GeoDataFrame
    persons_locations = persons_df[['person_id', lat_col, lon_col]].copy()
    persons_locations = persons_locations.dropna(subset=[lat_col, lon_col])
    persons_locations[lat_col] = pd.to_numeric(persons_locations[lat_col], errors='coerce')
    persons_locations[lon_col] = pd.to_numeric(persons_locations[lon_col], errors='coerce')
    persons_locations = persons_locations.dropna(subset=[lat_col, lon_col])

    if persons_locations.empty:
        logging.warning("No valid birth location coordinates found for mapping.")
        # Add empty placeholder columns to original df
        persons_df['birth_parish_code'] = None
        persons_df['birth_parish_name'] = None
        persons_df['birth_parish_is_western_line'] = None
        return persons_df

    try:
        persons_gdf = gpd.GeoDataFrame(
            persons_locations,
            geometry=gpd.points_from_xy(persons_locations[lon_col], persons_locations[lat_col]),
            crs="EPSG:4326"  # Assume WGS84
        )
    except Exception as e:
        logging.error(f"Error creating points GeoDataFrame: {e}")
        return persons_df

    # --- 3. Align CRS ---
    try:
        persons_gdf = persons_gdf.to_crs(parish_crs)
        logging.info(f"Points CRS transformed to {persons_gdf.crs}")
    except Exception as e:
        logging.error(f"Error transforming points CRS: {e}")
        return persons_df
        
    # --- 4. Spatial Join ---
    logging.info("Performing spatial join for birth locations...")
    try:
        # Keep only essential columns from parishes to avoid large merge
        parishes_simplified = parishes_gdf[['geometry', 'prsh_cd', 'prsh_nm', 'wstrn_l']].copy() 
        
        joined_gdf = gpd.sjoin(persons_gdf, parishes_simplified, how='left', predicate='within')
        logging.info(f"Spatial join complete. Found parish info for {joined_gdf['index_right'].notna().sum()} points.")

        # Handle potential duplicate matches if a point is in multiple polygons (unlikely with 'within', but good practice)
        # Keep the first match per person_id
        joined_gdf = joined_gdf.drop_duplicates(subset=['person_id'], keep='first')

    except Exception as e:
        logging.error(f"Error during spatial join: {e}")
        return persons_df

    # --- 5. Merge Results Back ---
    # Select columns to merge: person_id and the parish info
    # Adjust column names based on your shapefile columns used in parishes_simplified
    merge_cols = ['person_id', 'prsh_cd', 'prsh_nm', 'wstrn_l'] 
    cols_to_rename = { 
        'prsh_cd': 'birth_parish_code',    # Use 'prsh_cd' from shapefile
        'prsh_nm': 'birth_parish_name',    # Use 'prsh_nm' from shapefile
        'wstrn_l': 'birth_parish_is_western_line' # Use 'wstrn_l' from shapefile
    }
    
    # Select and rename columns from the *joined* GeoDataFrame
    mapping_results = joined_gdf[merge_cols].rename(columns=cols_to_rename)

    # Merge back into the original full persons DataFrame
    persons_df_mapped = pd.merge(persons_df, mapping_results, on='person_id', how='left')
    
    # Convert western_line status to boolean or integer if desired (it might be float/object after merge)
    if 'birth_parish_is_western_line' in persons_df_mapped.columns:
        # Assuming 1=WL, 0=Control. FillNA with -1
        persons_df_mapped['birth_parish_is_western_line'] = pd.to_numeric(persons_df_mapped['birth_parish_is_western_line'], errors='coerce').fillna(-1).astype(int) 
        logging.info(f"Mapped 'western_line' status distribution:\n{persons_df_mapped['birth_parish_is_western_line'].value_counts(dropna=False)}")

    logging.info("Finished spatial mapping.")
    return persons_df_mapped


# --- Main Execution ---

def main():
    """Main execution function."""
    logging.info(f"Starting data extraction and mapping process...")
    
    # 1. Extract data from JSON files into a list
    persons_data_list = extract_persons_data()
    
    if not persons_data_list:
        logging.warning("No relevant persons found or extracted. Exiting.")
        return 1
        
    # 2. Convert list to DataFrame
    persons_df = pd.DataFrame(persons_data_list)
    logging.info(f"Created initial DataFrame with {len(persons_df)} persons.")

    # 3. Perform Spatial Mapping for Birth Location
    # Check if shapefile exists before attempting mapping
    if os.path.exists(PARISH_SHAPEFILE_PATH):
         persons_df_mapped = map_birthplace_to_parish(persons_df, PARISH_SHAPEFILE_PATH)
    else:
         logging.error(f"Parish shapefile not found at {PARISH_SHAPEFILE_PATH}. Skipping spatial mapping.")
         # Add empty placeholder columns if mapping skipped
         persons_df['birth_parish_code'] = None
         persons_df['birth_parish_name'] = None
         persons_df['birth_parish_is_western_line'] = None
         persons_df_mapped = persons_df # Use the unmapped df

    # 4. Save the final DataFrame with mapping results
    output_csv_path = os.path.join(OUTPUT_DIR, 'persons_data_with_birth_parish.csv')
    persons_df_mapped.to_csv(output_csv_path, index=False, encoding='utf-8')
    logging.info(f"Saved final data for {len(persons_df_mapped)} persons (including birth parish info) to {output_csv_path}")
    
    # --- Optional: Perform other analyses (e.g., by decade) on the final mapped data ---
    
    # Analyze education by decade using the mapped data
    decade_stats = analyze_education_by_decade(persons_df_mapped.to_dict('records')) # Convert back to list of dicts for existing function
    if not decade_stats.empty:
         decade_output_path = os.path.join(OUTPUT_DIR, 'persons_education_by_decade.csv')
         decade_stats.to_csv(decade_output_path, index=False, encoding='utf-8')
         logging.info(f"Saved education analysis by decade to {decade_output_path}")
         # Optional: Print summary to console
         # print("\nEducation by Birth Decade Summary:")
         # print(decade_stats[['decade', 'total_persons', 'pct_technical', 'pct_business', 'pct_other_higher']])
    else:
         logging.info("Skipped saving decade analysis due to lack of data.")

    # Extract education metadata (can use the original list or the mapped df)
    education_metadata = extract_education_metadata(persons_df_mapped.to_dict('records'))
    metadata_output_path = os.path.join(OUTPUT_DIR, 'education_metadata.json')
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(education_metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved education metadata to {metadata_output_path}")
    
    logging.info("Processing finished.")
    return 0

if __name__ == "__main__":
    # Add geopandas dependency requirement check if desired
    try:
        import geopandas
    except ImportError:
        logging.error("Module 'geopandas' not found. Please install it (`pip install geopandas`) to run spatial mapping.")
        exit(1) # Exit if geopandas is missing
        
    main()