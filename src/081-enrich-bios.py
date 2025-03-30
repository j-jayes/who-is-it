import os
import json
import pandas as pd
from glob import glob
import logging
from typing import Dict, Any, Optional, List, Callable

# Set up logging
logging.basicConfig(
    filename='data/enrichment_process.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Define directories
INPUT_DIR = "data/structured_biographies"
OUTPUT_DIR = "data/enriched_biographies"
OCCUPATIONS_DIR = "data/occupations_for_classification/occupations_classified"
GEOCODE_CACHE_PATH = "data/locations/geocode_cache.json"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BiographyEnricher:
    """
    A modular class for enriching biographical data with various types of information.
    New enrichers can be added by creating methods and registering them.
    """
    
    def __init__(self):
        self.enrichers = []
        self.resources = {}
        logging.info("Initializing Biography Enricher")
    
    def add_enricher(self, name: str, function: Callable, resources: Dict[str, Any] = None):
        """Register a new enricher function with optional resources."""
        self.enrichers.append((name, function))
        if resources:
            self.resources[name] = resources
        logging.info(f"Added enricher: {name}")
    
    def enrich_biography(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all registered enrichers to the biography data."""
        enriched_data = data.copy()
        for name, enricher in self.enrichers:
            try:
                resources = self.resources.get(name, {})
                enriched_data = enricher(enriched_data, resources)
                logging.debug(f"Applied {name} enricher")
            except Exception as e:
                logging.error(f"Error applying {name} enricher: {e}")
        return enriched_data
    
    def process_all_biographies(self):
        """Process all biography files in the input directory."""
        files_processed = 0
        files_enriched = 0
        
        for filename in glob(os.path.join(INPUT_DIR, "*.json")):
            logging.info(f"Processing file: {os.path.basename(filename)}")
            files_processed += 1
            
            try:
                # Load the biography JSON
                with open(filename, 'r', encoding='utf-8') as f:
                    biography_data = json.load(f)
                
                # Process and enrich the biography
                enriched_biography = self.enrich_biography(biography_data)
                
                # Save the enriched biography
                output_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched_biography, f, ensure_ascii=False, indent=4)
                
                files_enriched += 1
                
            except Exception as e:
                logging.error(f"Error processing file {os.path.basename(filename)}: {e}")
        
        logging.info(f"Processing complete. {files_enriched}/{files_processed} files enriched.")


# ===== OCCUPATION ENRICHER =====

def load_occupation_resources():
    """Load resources needed for occupation enrichment."""
    try:
        english_classification = pd.read_parquet(
            os.path.join(OCCUPATIONS_DIR, "english_occupations_classified_joined.parquet")
        )
        swedish_classification = pd.read_parquet(
            os.path.join(OCCUPATIONS_DIR, "swedish_occupations_classified_joined.parquet")
        )
        
        # Convert to dictionaries for faster lookup
        english_dict = english_classification.set_index('occupation').to_dict('index')
        swedish_dict = swedish_classification.set_index('occupation').to_dict('index')
        
        logging.info("Occupation classification data loaded successfully.")
        return {
            "english_dict": english_dict,
            "swedish_dict": swedish_dict
        }
    except Exception as e:
        logging.error(f"Failed to load occupation classification data: {e}")
        raise e

def enrich_single_occupation(occupation_data, resources):
    """Enrich a single occupation record with HISCO codes."""
    if not occupation_data:
        return occupation_data
    
    english_dict = resources.get("english_dict", {})
    swedish_dict = resources.get("swedish_dict", {})
    
    # Enrich Swedish occupation
    swedish_title = occupation_data.get('occupation', '')
    if swedish_title and swedish_title in swedish_dict:
        classification = swedish_dict[swedish_title]
        if classification['prob_1'] > 0.75:
            occupation_data['hisco_code_swedish'] = classification['hisco_1']
            occupation_data['hisco_description_swedish'] = classification['desc_1']
    
    # Enrich English occupation
    english_title = occupation_data.get('occupation_english', '')
    if english_title and english_title in english_dict:
        classification = english_dict[english_title]
        if classification['prob_1'] > 0.75:
            occupation_data['hisco_code_english'] = classification['hisco_1']
            occupation_data['hisco_description_english'] = classification['desc_1']
    
    return occupation_data

def occupation_enricher(data, resources):
    """Enrich occupation data throughout the biography."""
    # Enrich main person's occupation
    if 'person' in data and 'occupation' in data['person']:
        occupation_data = data['person'].get('occupation', {})
        data['person']['occupation'] = enrich_single_occupation(occupation_data, resources)

    # Enrich parents' occupations
    if 'person' in data and 'parents' in data['person']:
        parents = data['person'].get('parents', [])
        if parents and isinstance(parents, list):
            for parent in parents:
                parent_occupation_data = parent.get('occupation', {})
                if parent_occupation_data:
                    parent['occupation'] = enrich_single_occupation(parent_occupation_data, resources)

    # Enrich spouse's occupation (if present)
    if 'family' in data and 'spouse' in data['family'] and data['family']['spouse']:
        spouse_occupation_data = data['family']['spouse'].get('occupation', {})
        if spouse_occupation_data:
            data['family']['spouse']['occupation'] = enrich_single_occupation(
                spouse_occupation_data, resources
            )

    # Enrich spouse's parents' occupations (if present)
    if ('family' in data and 'spouse' in data['family'] and data['family']['spouse'] and
            'parents' in data['family']['spouse']):
        spouse_parents = data['family']['spouse'].get('parents', [])
        if spouse_parents and isinstance(spouse_parents, list):
            for spouse_parent in spouse_parents:
                spouse_parent_occupation_data = spouse_parent.get('occupation', {})
                if spouse_parent_occupation_data:
                    spouse_parent['occupation'] = enrich_single_occupation(
                        spouse_parent_occupation_data, resources
                    )
    
    return data


# ===== LOCATION ENRICHER =====

def load_location_resources():
    """Load resources needed for location enrichment."""
    try:
        with open(GEOCODE_CACHE_PATH, 'r', encoding='utf-8') as f:
            geocode_cache = json.load(f)
        logging.info(f"Loaded geocode cache with {len(geocode_cache)} locations")
        return {"geocode_cache": geocode_cache}
    except Exception as e:
        logging.error(f"Error loading geocode cache: {e}")
        raise e

def enrich_single_location(location_value, resources):
    """Enrich a single location string with geocode data."""
    if not location_value or not isinstance(location_value, str):
        return location_value
    
    geocode_cache = resources.get("geocode_cache", {})
    
    # Look up the location in the geocode cache
    geocode_data = geocode_cache.get(location_value)
    if not geocode_data:
        return location_value
    
    # Create enriched location object
    return {
        "name": location_value,
        "latitude": geocode_data.get("latitude"),
        "longitude": geocode_data.get("longitude"),
        "formatted_address": geocode_data.get("formatted_address")
    }

def location_enricher(data, resources):
    """Enrich location data throughout the biography."""
    # Enrich career locations
    if 'career' in data and data['career']:
        for career_entry in data['career']:
            if career_entry.get('location'):
                career_entry['location'] = enrich_single_location(
                    career_entry['location'], resources
                )
    
    # Enrich birth place
    if 'person' in data and data['person'].get('birth_place'):
        data['person']['birth_place'] = enrich_single_location(
            data['person']['birth_place'], resources
        )
    
    # Enrich current location
    if data.get('current_location'):
        data['current_location'] = enrich_single_location(
            data['current_location'], resources
        )
    
    # Enrich spouse's birth place
    if ('family' in data and data['family'] and 
        'spouse' in data['family'] and data['family']['spouse'] and 
        data['family']['spouse'].get('birth_place')):
        data['family']['spouse']['birth_place'] = enrich_single_location(
            data['family']['spouse']['birth_place'], resources
        )
    
    # Enrich travel locations
    if 'travels' in data and data['travels']:
        for travel in data['travels']:
            if travel.get('country'):
                travel['country'] = enrich_single_location(
                    travel.get('country'), resources
                )
    
    return data


# ===== MAIN EXECUTION =====

def main():
    """Main execution function."""
    logging.info("Starting biography enrichment process")
    
    # Initialize the enricher
    enricher = BiographyEnricher()
    
    # Load resources and add enrichers
    try:
        # Add occupation enricher
        occupation_resources = load_occupation_resources()
        enricher.add_enricher("occupation", occupation_enricher, occupation_resources)
        
        # Add location enricher
        location_resources = load_location_resources()
        enricher.add_enricher("location", location_enricher, location_resources)
        
        # Process all biographies
        enricher.process_all_biographies()
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        return 1
    
    logging.info("Biography enrichment process completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())