import os
import json
import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime

def extract_decade(birth_date):
    """
    Extract decade from a birth date string.
    
    Args:
        birth_date (str): Birth date in format 'DD-MM-YYYY' or similar
        
    Returns:
        str: Decade (e.g., '1880s') or None if cannot be parsed
    """
    if not birth_date:
        return None
    
    # Try to extract year using regex (looking for 4 consecutive digits)
    year_match = re.search(r'(\d{4})', birth_date)
    if year_match:
        year = int(year_match.group(1))
    else:
        # Alternative format: try to parse DD-MM-YYYY
        try:
            parts = birth_date.split('-')
            if len(parts) == 3:
                year = int(parts[2])
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    # Calculate decade
    decade = (year // 10) * 10
    return f"{decade}s"

def analyze_occupations(directory_path):
    """
    Analyze occupations in biography JSON files by gender and decade of birth.
    
    Args:
        directory_path (str): Path to the directory containing biography JSON files.
        
    Returns:
        tuple: Contains DataFrames for male and female occupation statistics
    """
    # Dictionary to store occupation counts by gender and decade
    male_occupations = defaultdict(Counter)
    female_occupations = defaultdict(Counter)
    
    # Store HISCO codes and descriptions
    occupation_hisco_map = {}
    
    # Count total number of people with occupations by gender and decade
    male_decade_totals = Counter()
    female_decade_totals = Counter()
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            # Load the JSON data
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Extract person's information
                    if 'person' in data and data['person']:
                        person = data['person']
                        
                        # Skip if gender is not specified
                        if 'gender' not in person:
                            continue
                        
                        gender = person.get('gender')
                        birth_date = person.get('birth_date')
                        decade = extract_decade(birth_date)
                        
                        # Skip if we couldn't determine the decade
                        if not decade:
                            continue
                        
                        # Extract occupation
                        if 'occupation' in person and person['occupation']:
                            occupation_info = person['occupation']
                            occupation = occupation_info.get('occupation')
                            
                            if occupation:
                                # Standardize the case (convert to lowercase for counting)
                                occupation_standardized = occupation.lower()
                                
                                # Store HISCO code and description
                                # Keep original case for display purposes, but use standardized case for counting
                                hisco_code = occupation_info.get('hisco_code_swedish')
                                hisco_desc = occupation_info.get('hisco_description_swedish')
                                
                                # This is now handled in the counting section
                                
                                # Count the occupation using standardized case
                                if gender.lower() == 'male':
                                    male_occupations[decade][occupation_standardized] += 1
                                    male_decade_totals[decade] += 1
                                elif gender.lower() == 'female':
                                    female_occupations[decade][occupation_standardized] += 1
                                    female_decade_totals[decade] += 1
                                
                                # Map standardized occupation to original for display
                                occupation_hisco_map[occupation_standardized] = {
                                    'original': occupation,  # Store original for display
                                    'hisco_code': hisco_code,
                                    'hisco_description': hisco_desc
                                }
            
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Process male occupation data
    male_data = []
    for decade, occupations in male_occupations.items():
        decade_total = male_decade_totals[decade]
        for occupation, count in occupations.most_common(10):
            hisco_info = occupation_hisco_map.get(occupation, {})
            male_data.append({
                'Decade': decade,
                'Occupation': hisco_info.get('original', occupation),  # Use original case for display
                'Count': count,
                'Share': round(count / decade_total * 100, 2) if decade_total > 0 else 0,
                'HISCO Code': hisco_info.get('hisco_code', 'N/A'),
                'HISCO Description': hisco_info.get('hisco_description', 'N/A')
            })
    
    # Process female occupation data
    female_data = []
    for decade, occupations in female_occupations.items():
        decade_total = female_decade_totals[decade]
        for occupation, count in occupations.most_common(10):
            hisco_info = occupation_hisco_map.get(occupation, {})
            female_data.append({
                'Decade': decade,
                'Occupation': hisco_info.get('original', occupation),  # Use original case for display
                'Count': count,
                'Share': round(count / decade_total * 100, 2) if decade_total > 0 else 0,
                'HISCO Code': hisco_info.get('hisco_code', 'N/A'),
                'HISCO Description': hisco_info.get('hisco_description', 'N/A')
            })
    
    # Create DataFrames
    male_df = pd.DataFrame(male_data)
    female_df = pd.DataFrame(female_data)
    
    # Sort by decade and count
    if not male_df.empty:
        male_df = male_df.sort_values(['Decade', 'Count'], ascending=[True, False])
    if not female_df.empty:
        female_df = female_df.sort_values(['Decade', 'Count'], ascending=[True, False])
    
    return male_df, female_df

def analyze_overall_top_occupations(directory_path):
    """
    Analyze the overall top 10 occupations by gender across all decades.
    
    Args:
        directory_path (str): Path to the directory containing biography JSON files.
        
    Returns:
        tuple: Contains DataFrames for overall male and female occupation statistics
    """
    # Counters for all occupations
    male_all_occupations = Counter()
    female_all_occupations = Counter()
    total_males = 0
    total_females = 0
    
    # Store HISCO codes and descriptions
    occupation_hisco_map = {}
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            # Load the JSON data
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Extract person's information
                    if 'person' in data and data['person']:
                        person = data['person']
                        
                        # Skip if gender is not specified
                        if 'gender' not in person:
                            continue
                        
                        gender = person.get('gender')
                        
                        # Extract occupation
                        if 'occupation' in person and person['occupation']:
                            occupation_info = person['occupation']
                            occupation = occupation_info.get('occupation')
                            
                            if occupation:
                                # Store HISCO code and description
                                hisco_code = occupation_info.get('hisco_code_swedish')
                                hisco_desc = occupation_info.get('hisco_description_swedish')
                                
                                if hisco_code and hisco_desc:
                                    occupation_hisco_map[occupation] = {
                                        'hisco_code': hisco_code,
                                        'hisco_description': hisco_desc
                                    }
                                
                                # Standardize the case for overall counting too
                                occupation_standardized = occupation.lower()
                                
                                # Count the occupation
                                if gender.lower() == 'male':
                                    male_all_occupations[occupation_standardized] += 1
                                    total_males += 1
                                elif gender.lower() == 'female':
                                    female_all_occupations[occupation_standardized] += 1
                                    total_females += 1
                                
                                # Map standardized occupation to original for display
                                if hisco_code and hisco_desc:
                                    occupation_hisco_map[occupation_standardized] = {
                                        'original': occupation,  # Store original for display
                                        'hisco_code': hisco_code,
                                        'hisco_description': hisco_desc
                                    }
            
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Process top 10 male occupations
    male_top_data = []
    for occupation, count in male_all_occupations.most_common(10):
        hisco_info = occupation_hisco_map.get(occupation, {})
        male_top_data.append({
            'Occupation': hisco_info.get('original', occupation),  # Use original case for display
            'Count': count,
            'Share': round(count / total_males * 100, 2) if total_males > 0 else 0,
            'HISCO Code': hisco_info.get('hisco_code', 'N/A'),
            'HISCO Description': hisco_info.get('hisco_description', 'N/A')
        })
    
    # Process top 10 female occupations
    female_top_data = []
    for occupation, count in female_all_occupations.most_common(10):
        hisco_info = occupation_hisco_map.get(occupation, {})
        female_top_data.append({
            'Occupation': hisco_info.get('original', occupation),  # Use original case for display
            'Count': count,
            'Share': round(count / total_females * 100, 2) if total_females > 0 else 0,
            'HISCO Code': hisco_info.get('hisco_code', 'N/A'),
            'HISCO Description': hisco_info.get('hisco_description', 'N/A')
        })
    
    # Create DataFrames
    male_top_df = pd.DataFrame(male_top_data)
    female_top_df = pd.DataFrame(female_top_data)
    
    return male_top_df, female_top_df

def main():
    """Main function to run the occupation analysis."""
    # Directory path containing the biography JSON files
    directory_path = "data/enriched_biographies"
    
    # Run the analysis by decade
    male_occupations_by_decade, female_occupations_by_decade = analyze_occupations(directory_path)
    
    # Run the overall top 10 analysis
    male_top_occupations, female_top_occupations = analyze_overall_top_occupations(directory_path)
    
    # Display results
    print("=== Top 10 Male Occupations by Decade ===")
    print(male_occupations_by_decade)
    
    print("\n=== Top 10 Female Occupations by Decade ===")
    print(female_occupations_by_decade)
    
    print("\n=== Overall Top 10 Male Occupations ===")
    print(male_top_occupations)
    
    print("\n=== Overall Top 10 Female Occupations ===")
    print(female_top_occupations)
    
    # Save results to CSV files
    male_occupations_by_decade.to_csv("analysis/male_occupations_by_decade.csv", index=False)
    female_occupations_by_decade.to_csv("analysis/female_occupations_by_decade.csv", index=False)
    male_top_occupations.to_csv("analysis/male_top_occupations.csv", index=False)
    female_top_occupations.to_csv("analysis/female_top_occupations.csv", index=False)
    
    print("\nResults saved to CSV files.")

if __name__ == "__main__":
    main()