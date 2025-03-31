import os
import json
import pandas as pd
from glob import glob
import logging
from collections import defaultdict
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

# Educational institution classification keywords
TECHNICAL_KEYWORDS = [
    'tekniska', 'chalmers', 'kth', 'tekn', 'ingenjör', 'teknisk', 
    'teknolog', 'polytekn', 'engineering', 'technical'
]

BUSINESS_KEYWORDS = [
    'handels', 'ekonomi', 'handelshögskola', 'business', 'commerce', 
    'ekonomisk', 'handelsinstitut', 'handelsgymnasium', 'ekonom'
]

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

def classify_education(education_entries: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Classify education entries by institution type."""
    if not education_entries:
        return {"technical": False, "business": False, "other": False}
    
    result = {"technical": False, "business": False, "other": False}
    
    for entry in education_entries:
        institution = entry.get("institution", "").lower()
        degree = entry.get("degree", "").lower()
        
        # Check if any technical keywords appear in institution or degree
        if any(keyword.lower() in institution or keyword.lower() in degree for keyword in TECHNICAL_KEYWORDS):
            result["technical"] = True
        
        # Check if any business keywords appear in institution or degree
        elif any(keyword.lower() in institution or keyword.lower() in degree for keyword in BUSINESS_KEYWORDS):
            result["business"] = True
        
        # If it's a higher education but not classified yet, mark as other
        elif entry.get("degree_level") in ["Master's", "PhD", "Bachelor's"]:
            result["other"] = True
    
    return result

def is_director(person_data: Dict[str, Any]) -> bool:
    """Check if the person is a director based on HISCO code."""
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

def extract_directors_education():
    """Extract educational information for directors."""
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
            person = data.get('person', {})
            if person and is_director(person):
                directors_count += 1
                
                education = data.get('education', []) or []
                
                # Extract relevant information
                birth_date = person.get('birth_date')
                birth_decade = extract_birth_decade(birth_date)
                education_classification = classify_education(education)
                
                # Create a record
                occupation = person.get('occupation', {}) or {}
                director_record = {
                    'id': os.path.basename(filename),
                    'name': f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                    'first_name': person.get('first_name', ''),
                    'middle_name': person.get('middle_name', ''),
                    'last_name': person.get('last_name', ''),
                    'birth_date': birth_date,
                    'birth_decade': birth_decade,
                    'occupation': occupation.get('occupation'),
                    'hisco_code_swedish': occupation.get('hisco_code_swedish'),
                    'hisco_code_english': occupation.get('hisco_code_english'),
                    'has_technical_education': education_classification['technical'],
                    'has_business_education': education_classification['business'],
                    'has_other_higher_education': education_classification['other'],
                    'education': education
                }
                
                directors_data.append(director_record)
                
        except Exception as e:
            logging.error(f"Error processing file {os.path.basename(filename)}: {e}")
    
    logging.info(f"Completed processing {files_processed} files, found {directors_count} directors")
    return directors_data

def analyze_education_by_decade(directors_data: List[Dict[str, Any]]):
    """Analyze education distribution by birth decade."""
    # Create a DataFrame
    df = pd.DataFrame(directors_data)
    
    # Filter out directors with missing birth decade
    df_with_decade = df.dropna(subset=['birth_decade'])
    
    # Group by birth decade
    decade_stats = df_with_decade.groupby('birth_decade').agg({
        'id': 'count',
        'has_technical_education': 'sum',
        'has_business_education': 'sum',
        'has_other_higher_education': 'sum'
    }).reset_index()
    
    # Calculate percentages
    decade_stats['total_directors'] = decade_stats['id']
    decade_stats['pct_technical'] = (decade_stats['has_technical_education'] / decade_stats['total_directors'] * 100).round(1)
    decade_stats['pct_business'] = (decade_stats['has_business_education'] / decade_stats['total_directors'] * 100).round(1)
    decade_stats['pct_other_higher'] = (decade_stats['has_other_higher_education'] / decade_stats['total_directors'] * 100).round(1)
    
    # Clean up column names
    decade_stats = decade_stats.rename(columns={
        'birth_decade': 'decade',
        'id': 'total_count'
    })
    
    return decade_stats

def extract_education_metadata(directors_data: List[Dict[str, Any]]):
    """Extract metadata about education institutions for manual review."""
    education_institutions = defaultdict(int)
    education_degrees = defaultdict(int)
    
    for director in directors_data:
        # Make sure education is a list and not None
        education_entries = director.get('education', []) or []
        
        for edu in education_entries:
            if 'institution' in edu and edu['institution']:
                education_institutions[edu['institution']] += 1
            if 'degree' in edu and edu['degree']:
                education_degrees[edu['degree']] += 1
    
    # Convert to sorted lists of dicts for easier viewing
    institutions_list = [{'institution': k, 'count': v} for k, v in education_institutions.items()]
    institutions_list = sorted(institutions_list, key=lambda x: x['count'], reverse=True)
    
    degrees_list = [{'degree': k, 'count': v} for k, v in education_degrees.items()]
    degrees_list = sorted(degrees_list, key=lambda x: x['count'], reverse=True)
    
    return {
        'institutions': institutions_list,
        'degrees': degrees_list
    }

def main():
    """Main execution function."""
    logging.info("Starting director education analysis")
    
    # Extract data from all director biographies
    directors_data = extract_directors_education()
    
    # Save the raw data
    directors_df = pd.DataFrame(directors_data)
    directors_df.to_csv(os.path.join(OUTPUT_DIR, 'all_education_data.csv'), index=False)
    
    # Perform analysis by decade
    decade_stats = analyze_education_by_decade(directors_data)
    
    # Save the decade analysis
    decade_stats.to_csv(os.path.join(OUTPUT_DIR, 'all_education_by_decade.csv'), index=False)
    
    # Extract education metadata for classification refinement
    education_metadata = extract_education_metadata(directors_data)
    
    # Save education metadata
    with open(os.path.join(OUTPUT_DIR, 'all_education_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(education_metadata, f, ensure_ascii=False, indent=2)
    
    # Print summary
    logging.info(f"Found {len(directors_data)} directors with HISCO codes 21000-21900")
    logging.info(f"Analysis saved to {OUTPUT_DIR}")
    logging.info("\nEducation by Birth Decade:")
    print(decade_stats[['decade', 'total_count', 'pct_technical', 'pct_business', 'pct_other_higher']])
    
    return 0

if __name__ == "__main__":
    exit(main())