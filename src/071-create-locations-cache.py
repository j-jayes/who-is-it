import os
import json
import pandas as pd
from glob import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define input and output directories
input_dir = "data/structured_biographies"
output_dir = "data/locations"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize list to store all locations
locations = []

# Process each JSON file in the input directory
for filename in glob(os.path.join(input_dir, "*.json")):
    logging.info(f"Processing file: {os.path.basename(filename)}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract locations from career entries (only location field, not organization)
        if 'career' in data and data['career']:
            for career_entry in data['career']:
                # Include location even if country_code is missing
                if career_entry.get('location'):
                    locations.append({
                        'location': career_entry.get('location'),
                        'country_code': career_entry.get('country_code'),  # This might be None/null
                        'source_type': 'career',
                        'source_file': os.path.basename(filename)
                    })
        
        # Extract person's birth place
        if 'person' in data and data['person'].get('birth_place'):
            locations.append({
                'location': data['person']['birth_place'],
                'country_code': None,  # Don't assume any country code
                'source_type': 'birth_place',
                'source_file': os.path.basename(filename)
            })
        
        # Extract current location
        if data.get('current_location'):
            locations.append({
                'location': data['current_location'],
                'country_code': None,  # Don't assume any country code
                'source_type': 'current_location',
                'source_file': os.path.basename(filename)
            })
        
        # Extract spouse's birth place
        if ('family' in data and data['family'] and 'spouse' in data['family'] and 
            data['family']['spouse'] and data['family']['spouse'].get('birth_place')):
            locations.append({
                'location': data['family']['spouse']['birth_place'],
                'country_code': None,  # Don't assume any country code
                'source_type': 'spouse_birth_place',
                'source_file': os.path.basename(filename)
            })
        
        # Extract travel locations
        if 'travels' in data and data['travels']:
            for travel in data['travels']:
                if travel.get('country'):
                    locations.append({
                        'location': travel.get('country'),
                        'country_code': travel.get('country_code'),  # This might be None/null
                        'source_type': 'travel',
                        'source_file': os.path.basename(filename)
                    })
                
    except Exception as e:
        logging.error(f"Error processing file {os.path.basename(filename)}: {e}")

# Create DataFrame from the collected locations
df_locations = pd.DataFrame(locations)

# Remove duplicates but keep track of all sources
# Group by location and country_code, and aggregate the sources
df_locations_agg = df_locations.groupby(['location', 'country_code'], dropna=False).agg({
    'source_type': lambda x: list(set(x)),
    'source_file': lambda x: list(set(x))
}).reset_index()

# Add count of appearances
df_locations_agg['count'] = df_locations_agg['source_file'].apply(len)

# Sort by frequency of appearance
df_locations_agg = df_locations_agg.sort_values('count', ascending=False)

# Save to parquet file
output_path = os.path.join(output_dir, 'locations_cache.parquet')

try:
    df_locations_agg.to_parquet(output_path)
    logging.info(f"Locations cache saved to {output_path}")
    logging.info(f"Total unique locations: {len(df_locations_agg)}")
except Exception as e:
    logging.error(f"Error saving locations cache: {e}")

# Also save as CSV for easier inspection
csv_output_path = os.path.join(output_dir, 'locations_cache.csv')
try:
    df_locations_agg.to_csv(csv_output_path, index=False)
    logging.info(f"Locations cache also saved as CSV to {csv_output_path}")
except Exception as e:
    logging.error(f"Error saving locations CSV: {e}")

# Print some stats
logging.info(f"Top 10 most frequent locations:")
for i, (idx, row) in enumerate(df_locations_agg.head(10).iterrows()):
    country_info = f" ({row['country_code']})" if pd.notna(row['country_code']) else ""
    logging.info(f"{i+1}. {row['location']}{country_info}: {row['count']} occurrences")

# Print stats about locations with and without country codes
with_country = df_locations_agg['country_code'].notna().sum()
without_country = df_locations_agg['country_code'].isna().sum()
logging.info(f"Locations with country code: {with_country}")
logging.info(f"Locations without country code: {without_country}")
logging.info(f"Percentage with country code: {(with_country / len(df_locations_agg) * 100):.1f}%")