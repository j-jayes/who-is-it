import os
import json
import pandas as pd
from glob import glob
import logging
from typing import Dict, List, Any, Optional
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories
INPUT_DIR = "data/enriched_biographies"
OUTPUT_DIR = "data/analysis"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_director(person_data: Dict[str, Any]) -> bool:
    """Check if the person is a director based on HISCO code (21000-21900)."""
    occupation = person_data.get("occupation", {})
    
    # Check Swedish HISCO code
    swedish_code = occupation.get("hisco_code_swedish")
    if swedish_code and isinstance(swedish_code, str):
        try:
            code_value = int(float(swedish_code))
            if 1000 <= code_value <= 99900:
                return True
        except (ValueError, TypeError):
            pass
    
    # Check English HISCO code
    english_code = occupation.get("hisco_code_english")
    if english_code and isinstance(english_code, str):
        try:
            code_value = int(float(english_code))
            if 1000 <= code_value <= 99900:
                return True
        except (ValueError, TypeError):
            pass
    
    return False

def extract_birth_decade(birth_date: str) -> Optional[int]:
    """Extract the decade from a birth date string."""
    if not birth_date:
        return None
    
    # Try to extract year from different date formats
    year_match = re.search(r'(\d{4})', birth_date)
    if year_match:
        year = int(year_match.group(1))
    else:
        # Try DD-MM-YYYY format
        parts = birth_date.split('-')
        if len(parts) == 3 and len(parts[2]) == 4:
            year = int(parts[2])
        else:
            return None
    
    # Calculate decade
    return (year // 10) * 10

def check_international_experience(career_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check career entries for international experience.
    Returns counts and details of USA and non-Swedish experiences.
    """
    result = {
        "usa_experience_count": 0,
        "non_swedish_experience_count": 0,
        "usa_positions": [],
        "non_swedish_positions": []
    }
    
    if not career_entries:
        return result
    
    for entry in career_entries:
        # Initialize flags
        is_usa = False
        is_non_swedish = False
        
        # Check country_code field
        country_code = entry.get("country_code")
        if country_code == "USA":
            is_usa = True
            is_non_swedish = True
        elif country_code and country_code != "SWE":
            is_non_swedish = True
        
        # Check location field
        location = entry.get("location", {})
        if isinstance(location, dict):
            formatted_address = location.get("formatted_address", "")
            if formatted_address:
                if "USA" in formatted_address:
                    is_usa = True
                    is_non_swedish = True
                elif "Sweden" not in formatted_address:
                    is_non_swedish = True
        
        # Update counts and details
        if is_usa:
            result["usa_experience_count"] += 1
            position_info = {
                "position": entry.get("position", ""),
                "organization": entry.get("organization", ""),
                "start_year": entry.get("start_year", ""),
                "end_year": entry.get("end_year", ""),
                "location": formatted_address if isinstance(location, dict) else location
            }
            result["usa_positions"].append(position_info)
        
        if is_non_swedish:
            result["non_swedish_experience_count"] += 1
            position_info = {
                "position": entry.get("position", ""),
                "organization": entry.get("organization", ""),
                "start_year": entry.get("start_year", ""),
                "end_year": entry.get("end_year", ""),
                "location": formatted_address if isinstance(location, dict) else location
            }
            result["non_swedish_positions"].append(position_info)
    
    return result

def extract_directors_international_experience():
    """Extract international experience information for directors."""
    directors_data = []
    files_processed = 0
    directors_count = 0
    
    # Process each JSON file in the input directory
    for filename in glob(os.path.join(INPUT_DIR, "*.json")):
        files_processed += 1
        
        if files_processed % 100 == 0:
            logging.info(f"Processed {files_processed} files, found {directors_count} directors")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the person is a director
            if 'person' in data and is_director(data['person']):
                directors_count += 1
                
                person = data['person']
                career = data.get('career', [])
                
                # Check for international experience
                international_exp = check_international_experience(career)
                
                # Extract birth decade
                birth_date = person.get('birth_date')
                birth_decade = extract_birth_decade(birth_date)
                
                # Create a record
                director_record = {
                    'id': os.path.basename(filename),
                    'name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                    'first_name': person.get('first_name', ''),
                    'middle_name': person.get('middle_name', ''),
                    'last_name': person.get('last_name', ''),
                    'birth_date': birth_date,
                    'birth_decade': birth_decade,
                    'occupation': person.get('occupation', {}).get('occupation'),
                    'hisco_code_swedish': person.get('occupation', {}).get('hisco_code_swedish'),
                    'usa_experience_count': international_exp['usa_experience_count'],
                    'non_swedish_experience_count': international_exp['non_swedish_experience_count'],
                    'has_usa_experience': international_exp['usa_experience_count'] > 0,
                    'has_non_swedish_experience': international_exp['non_swedish_experience_count'] > 0,
                    'usa_positions': international_exp['usa_positions'],
                    'non_swedish_positions': international_exp['non_swedish_positions']
                }
                
                directors_data.append(director_record)
                
        except Exception as e:
            logging.error(f"Error processing file {os.path.basename(filename)}: {e}")
    
    logging.info(f"Completed processing {files_processed} files, found {directors_count} directors")
    return directors_data

def analyze_experience_by_decade(directors_data: List[Dict[str, Any]]):
    """Analyze international experience distribution by birth decade."""
    # Create a DataFrame
    df = pd.DataFrame(directors_data)
    
    # Create a cleaner version for analysis (without the detailed position lists)
    analysis_df = df.drop(columns=['usa_positions', 'non_swedish_positions'], errors='ignore')
    
    # Filter out directors with missing birth decade
    df_with_decade = analysis_df.dropna(subset=['birth_decade'])
    
    # Group by birth decade
    decade_stats = df_with_decade.groupby('birth_decade').agg({
        'id': 'count',
        'has_usa_experience': 'sum',
        'has_non_swedish_experience': 'sum'
    }).reset_index()
    
    # Calculate percentages
    decade_stats['total_directors'] = decade_stats['id']
    decade_stats['pct_usa_experience'] = (decade_stats['has_usa_experience'] / decade_stats['total_directors'] * 100).round(1)
    decade_stats['pct_non_swedish_experience'] = (decade_stats['has_non_swedish_experience'] / decade_stats['total_directors'] * 100).round(1)
    
    # Clean up column names
    decade_stats = decade_stats.rename(columns={
        'birth_decade': 'decade',
        'id': 'total_count'
    })
    
    return decade_stats

def extract_international_locations(directors_data: List[Dict[str, Any]]):
    """Extract metadata about international locations for reference."""
    usa_locations = {}
    non_swedish_locations = {}
    
    for director in directors_data:
        # Extract USA locations
        for position in director.get('usa_positions', []):
            location = position.get('location', '')
            if location:
                if location not in usa_locations:
                    usa_locations[location] = 0
                usa_locations[location] += 1
                
        # Extract non-Swedish locations
        for position in director.get('non_swedish_positions', []):
            location = position.get('location', '')
            if location:
                if location not in non_swedish_locations:
                    non_swedish_locations[location] = 0
                non_swedish_locations[location] += 1
    
    # Convert to sorted lists
    usa_locations_list = [{'location': k, 'count': v} for k, v in usa_locations.items()]
    usa_locations_list = sorted(usa_locations_list, key=lambda x: x['count'], reverse=True)
    
    non_swedish_locations_list = [{'location': k, 'count': v} for k, v in non_swedish_locations.items()]
    non_swedish_locations_list = sorted(non_swedish_locations_list, key=lambda x: x['count'], reverse=True)
    
    return {
        'usa_locations': usa_locations_list,
        'non_swedish_locations': non_swedish_locations_list
    }

def main():
    """Main execution function."""
    logging.info("Starting director international experience analysis")
    
    # Extract data from all director biographies
    directors_data = extract_directors_international_experience()
    
    # Create a simplified version for CSV export (without the detailed position lists)
    export_data = []
    for director in directors_data:
        export_record = director.copy()
        
        # Convert position lists to counts only
        export_record.pop('usa_positions', None)
        export_record.pop('non_swedish_positions', None)
        
        export_data.append(export_record)
    
    # Save the data to CSV
    directors_df = pd.DataFrame(export_data)
    directors_df.to_csv(os.path.join(OUTPUT_DIR, 'all_international_experience.csv'), index=False)
    
    # Save the full data (including position details) to JSON
    with open(os.path.join(OUTPUT_DIR, 'all_international_experience_full.json'), 'w', encoding='utf-8') as f:
        json.dump(directors_data, f, ensure_ascii=False, indent=2)
    
    # Perform analysis by decade
    decade_stats = analyze_experience_by_decade(directors_data)
    
    # Save the decade analysis
    decade_stats.to_csv(os.path.join(OUTPUT_DIR, 'all_international_experience_by_decade.csv'), index=False)
    
    # Extract location metadata
    location_metadata = extract_international_locations(directors_data)
    
    # Save location metadata
    with open(os.path.join(OUTPUT_DIR, 'all_international_locations_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(location_metadata, f, ensure_ascii=False, indent=2)
    
    # Print summary
    logging.info(f"Found {len(directors_data)} directors with HISCO codes 21000-21900")
    logging.info(f"Analysis saved to {OUTPUT_DIR}")
    
    # Count directors with international experience
    usa_exp_count = sum(1 for d in directors_data if d['has_usa_experience'])
    non_swedish_exp_count = sum(1 for d in directors_data if d['has_non_swedish_experience'])
    
    logging.info(f"Directors with USA experience: {usa_exp_count} ({usa_exp_count/len(directors_data)*100:.1f}%)")
    logging.info(f"Directors with non-Swedish experience: {non_swedish_exp_count} ({non_swedish_exp_count/len(directors_data)*100:.1f}%)")
    
    logging.info("\nInternational Experience by Birth Decade:")
    print(decade_stats[['decade', 'total_count', 'pct_usa_experience', 'pct_non_swedish_experience']])
    
    return 0

if __name__ == "__main__":
    exit(main())