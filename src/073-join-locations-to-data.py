import os
import json
import logging
from glob import glob
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define input and output directories
geocode_cache_path = "data/locations/geocode_cache.json"
input_dir = "data/enriched_biographies_with_hisco_codes"
output_dir = "data/enriched_biographies_with_locations"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load geocode cache
try:
    with open(geocode_cache_path, 'r', encoding='utf-8') as f:
        geocode_cache = json.load(f)
    logging.info(f"Loaded geocode cache with {len(geocode_cache)} locations")
except Exception as e:
    logging.error(f"Error loading geocode cache: {e}")
    exit(1)

def enrich_location(location_value: str) -> Optional[Dict[str, Any]]:
    """Enrich a location string with geocode data if available."""
    if not location_value or not isinstance(location_value, str):
        return None
    
    # Look up the location in the geocode cache
    geocode_data = geocode_cache.get(location_value)
    if not geocode_data:
        logging.debug(f"No geocode data found for location: {location_value}")
        return None
    
    # Create enriched location object
    return {
        "name": location_value,
        "latitude": geocode_data.get("latitude"),
        "longitude": geocode_data.get("longitude"),
        "formatted_address": geocode_data.get("formatted_address")
    }

def process_biography(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a biography JSON and add geocode information to location fields."""
    # Create a deep copy to avoid modifying the original
    enriched_data = data.copy()
    
    # Enrich career locations
    if 'career' in enriched_data and enriched_data['career']:
        for career_entry in enriched_data['career']:
            if career_entry.get('location'):
                location_data = enrich_location(career_entry['location'])
                if location_data:
                    # Replace string with object containing original name and geocode info
                    career_entry['location'] = location_data
    
    # Enrich birth place
    if 'person' in enriched_data and enriched_data['person'].get('birth_place'):
        birth_place_data = enrich_location(enriched_data['person']['birth_place'])
        if birth_place_data:
            enriched_data['person']['birth_place'] = birth_place_data
    
    # Enrich current location
    if enriched_data.get('current_location'):
        current_location_data = enrich_location(enriched_data['current_location'])
        if current_location_data:
            enriched_data['current_location'] = current_location_data
    
    # Enrich spouse's birth place
    if ('family' in enriched_data and enriched_data['family'] and 
        'spouse' in enriched_data['family'] and enriched_data['family']['spouse'] and 
        enriched_data['family']['spouse'].get('birth_place')):
        spouse_birth_place_data = enrich_location(enriched_data['family']['spouse']['birth_place'])
        if spouse_birth_place_data:
            enriched_data['family']['spouse']['birth_place'] = spouse_birth_place_data
    
    # Enrich travel locations
    if 'travels' in enriched_data and enriched_data['travels']:
        for travel in enriched_data['travels']:
            if travel.get('country'):
                country_data = enrich_location(travel.get('country'))
                if country_data:
                    travel['country'] = country_data
    
    return enriched_data

# Process each JSON file in the input directory
files_processed = 0
files_enriched = 0

for filename in glob(os.path.join(input_dir, "*.json")):
    logging.info(f"Processing file: {os.path.basename(filename)}")
    files_processed += 1
    
    try:
        # Load the biography JSON
        with open(filename, 'r', encoding='utf-8') as f:
            biography_data = json.load(f)
        
        # Process and enrich the biography
        enriched_biography = process_biography(biography_data)
        
        # Save the enriched biography
        output_path = os.path.join(output_dir, os.path.basename(filename))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_biography, f, ensure_ascii=False, indent=2)
        
        files_enriched += 1
        
    except Exception as e:
        logging.error(f"Error processing file {os.path.basename(filename)}: {e}")

logging.info(f"Processing complete. {files_enriched}/{files_processed} files enriched with location data.")