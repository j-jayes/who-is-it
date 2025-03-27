import json
import os
import time
import pandas as pd
from pathlib import Path
import requests
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/locations/geocoding.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# File paths
LOCATIONS_CACHE_PATH = 'data/locations/locations_cache.parquet'
GEOCODE_CACHE_PATH = 'data/locations/geocode_cache.json'
GEOCODED_OUTPUT_PATH = 'data/locations/geocoded_locations.parquet'
FAILED_GEOCODES_PATH = 'data/locations/failed_geocodes.csv'

# TESTING LIMIT - Process only this many locations
# Set to 0 to process all locations
TESTING_LIMIT = 10000

# Google Maps Geocoding API key from .env
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not API_KEY:
    logging.error("GOOGLE_MAPS_API_KEY not found in environment variables")
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")

# Load or initialize the geocode cache
def load_geocode_cache(file_path):
    if Path(file_path).is_file():
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}

def save_geocode_cache(cache, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)

# Simple geocoding function
def geocode(address, api_key):
    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': address, 'key': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                result = data['results'][0]
                location = result['geometry']['location']
                
                # Extract more details from the result
                formatted_address = result.get('formatted_address', '')
                place_types = result.get('types', [])
                
                # Get administrative divisions if available
                admin_areas = {}
                for component in result.get('address_components', []):
                    for comp_type in component.get('types', []):
                        if comp_type.startswith('administrative_area_level_'):
                            admin_areas[comp_type] = component.get('long_name')
                
                logging.info(f"Geocoded: '{address}' → {location['lat']}, {location['lng']}")
                
                return {
                    'latitude': location['lat'],
                    'longitude': location['lng'],
                    'formatted_address': formatted_address,
                    'place_types': place_types,
                    'admin_areas': admin_areas,
                    'status': 'OK'
                }
            else:
                logging.warning(f"Failed to geocode '{address}': {data.get('status')}")
                return {
                    'status': data.get('status'),
                    'error_message': data.get('error_message', '')
                }
        else:
            logging.error(f"HTTP error for '{address}': {response.status_code}")
            return {
                'status': 'HTTP_ERROR',
                'error_message': f"HTTP {response.status_code}"
            }
    except Exception as e:
        logging.error(f"Exception geocoding '{address}': {str(e)}")
        return {
            'status': 'EXCEPTION',
            'error_message': str(e)
        }

# Main script
def main():
    # Load existing geocode cache
    geocode_cache = load_geocode_cache(GEOCODE_CACHE_PATH)
    
    # Load input locations from Parquet file
    try:
        df_locations = pd.read_parquet(LOCATIONS_CACHE_PATH)
        logging.info(f"Loaded {len(df_locations)} locations from {LOCATIONS_CACHE_PATH}")
    except Exception as e:
        logging.error(f"Failed to load locations from {LOCATIONS_CACHE_PATH}: {e}")
        return
    
    # Create a results DataFrame with the same index as df_locations
    df_results = pd.DataFrame(index=df_locations.index)
    df_results['location'] = df_locations['location']
    df_results['country_code'] = df_locations['country_code']
    df_results['source_types'] = df_locations['source_type']
    df_results['count'] = df_locations['count']
    
    # Create new columns for geocoding results
    df_results['latitude'] = None
    df_results['longitude'] = None
    df_results['formatted_address'] = None
    df_results['status'] = None
    df_results['admin_areas'] = None
    df_results['place_types'] = None
    
    # Failed geocodes list
    failed_geocodes = []
    
    # Process each location - sort by count (frequency) to process most common locations first
    sorted_results = df_results.sort_values('count', ascending=False)
    
    # Apply testing limit
    if TESTING_LIMIT > 0:
        logging.info(f"TESTING MODE: Processing only {TESTING_LIMIT} locations")
        sorted_results = sorted_results.head(TESTING_LIMIT)
    
    total_locations = len(sorted_results)
    cached_count = 0
    success_count = 0
    fail_count = 0
    
    for idx, row in sorted_results.iterrows():
        location = row['location']
        
        # Skip null/empty locations
        if pd.isna(location) or location.strip() == '':
            logging.warning(f"Skipping empty location at index {idx}")
            continue
        
        # Log progress
        logging.info(f"Processing {cached_count + success_count + fail_count + 1}/{total_locations}: '{location}'")
        
        if location in geocode_cache:
            logging.info(f"Using cached data for '{location}'")
            cached_count += 1
            result = geocode_cache[location]
        else:
            # Add a small delay to respect API rate limits
            time.sleep(0.2)
            
            # Geocode using the simple function
            result = geocode(location, API_KEY)
            
            # Cache the result regardless of success or failure
            geocode_cache[location] = result
            
            # Save updated cache every 5 requests in testing mode
            if (cached_count + success_count + fail_count) % 5 == 0:
                save_geocode_cache(geocode_cache, GEOCODE_CACHE_PATH)
        
        # Update the results DataFrame
        if result.get('status') == 'OK':
            df_results.at[idx, 'latitude'] = result.get('latitude')
            df_results.at[idx, 'longitude'] = result.get('longitude')
            df_results.at[idx, 'formatted_address'] = result.get('formatted_address')
            df_results.at[idx, 'status'] = result.get('status')
            df_results.at[idx, 'admin_areas'] = str(result.get('admin_areas', {}))
            df_results.at[idx, 'place_types'] = str(result.get('place_types', []))
            success_count += 1
        else:
            df_results.at[idx, 'status'] = result.get('status')
            fail_count += 1
            failed_geocodes.append({
                'location': location,
                'country_code': row['country_code'],
                'status': result.get('status'),
                'error_message': result.get('error_message', '')
            })
    
    # Save the final cache
    save_geocode_cache(geocode_cache, GEOCODE_CACHE_PATH)
    
    # Save the results - only save the processed rows in testing mode
    if TESTING_LIMIT > 0:
        # Only save rows that were processed (have a status)
        processed_results = df_results[df_results['status'].notna()]
        processed_results.to_parquet(GEOCODED_OUTPUT_PATH)
        logging.info(f"TEST results: {len(processed_results)} geocoded locations saved to {GEOCODED_OUTPUT_PATH}")
    else:
        df_results.to_parquet(GEOCODED_OUTPUT_PATH)
        logging.info(f"Geocoded locations saved to {GEOCODED_OUTPUT_PATH}")
    
    # Save failed geocodes
    pd.DataFrame(failed_geocodes).to_csv(FAILED_GEOCODES_PATH, index=False)
    logging.info(f"Failed geocodes saved to {FAILED_GEOCODES_PATH}")
    
    # Log summary
    logging.info("=== Geocoding Summary ===")
    logging.info(f"Testing limit: {TESTING_LIMIT if TESTING_LIMIT > 0 else 'None'}")
    logging.info(f"Total locations processed: {total_locations}")
    logging.info(f"Cached results: {cached_count}")
    logging.info(f"Successful geocodes: {success_count}")
    logging.info(f"Failed geocodes: {fail_count}")
    if success_count + fail_count > 0:
        logging.info(f"Success rate: {(success_count / (success_count + fail_count)) * 100:.1f}%")
    logging.info("=========================")

if __name__ == "__main__":
    main()