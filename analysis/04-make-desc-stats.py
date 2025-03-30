import os
import json
import pandas as pd
from collections import Counter, defaultdict
import re

def analyze_biographies(directory_path):
    """
    Analyze biography JSON files in the given directory.
    
    Args:
        directory_path (str): Path to the directory containing biography JSON files.
        
    Returns:
        tuple: Contains three elements:
            - DataFrame with book counts
            - DataFrame with field presence statistics
            - DataFrame with entry counts for non-empty fields
    """
    # Dictionaries to store our statistics
    book_counts = Counter()
    field_presence = defaultdict(int)
    field_entry_counts = defaultdict(list)
    
    # Total number of files
    total_files = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            total_files += 1
            
            # Extract book identifier (text before first underscore)
            match = re.match(r'^([^_]+)_', filename)
            if match:
                book_id = match.group(1)
                book_counts[book_id] += 1
            
            # Load the JSON data
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    
                    # Check presence of each top-level field and count entries
                    for field, value in data.items():
                        # Count presence of non-null fields
                        if value is not None:
                            field_presence[field] += 1
                            
                            # Count number of entries for list fields
                            if isinstance(value, list):
                                field_entry_counts[field].append(len(value))
                            # For dictionaries, count as 1 entry
                            elif isinstance(value, dict) and value:
                                field_entry_counts[field].append(1)
                            # For non-empty non-collection fields, count as 1
                            elif value:
                                field_entry_counts[field].append(1)
                            else:
                                field_entry_counts[field].append(0)
                        else:
                            field_entry_counts[field].append(0)
                            
                except json.JSONDecodeError:
                    print(f"Error parsing {filename}")
                    continue
    
    # Create DataFrame for book counts
    book_df = pd.DataFrame({
        'Book ID': list(book_counts.keys()),
        'Count': list(book_counts.values())
    }).sort_values('Count', ascending=False)
    
    # Calculate field presence statistics
    field_stats = []
    for field, count in field_presence.items():
        field_stats.append({
            'Field': field,
            'Present Count': count,
            'Missing Count': total_files - count,
            'Present Percentage': round((count / total_files) * 100, 2),
            'Missing Percentage': round(((total_files - count) / total_files) * 100, 2)
        })
    
    field_df = pd.DataFrame(field_stats).sort_values('Present Percentage', ascending=False)
    
    # Calculate entry count statistics for non-empty fields
    entry_stats = []
    for field, counts in field_entry_counts.items():
        if counts:
            mean_entries = sum(counts) / total_files
            no_info_percentage = (counts.count(0) / total_files) * 100
            entry_stats.append({
                'Field': field,
                'Mean Entries': round(mean_entries, 2),
                'Share with No Information (%)': round(no_info_percentage, 2)
            })
    
    entry_df = pd.DataFrame(entry_stats).sort_values('Mean Entries', ascending=False)
    
    return book_df, field_df, entry_df

def main():
    """Main function to run the analysis."""
    # Directory path containing the biography JSON files
    directory_path = "data/enriched_biographies"
    
    # Run the analysis
    book_counts, field_stats, entry_stats = analyze_biographies(directory_path)
    
    # Display results
    print("=== Book Counts ===")
    print(book_counts)
    print("\n=== Field Presence Statistics ===")
    print(field_stats)
    print("\n=== Field Entry Statistics ===")
    print(entry_stats)
    
    # Save results to CSV files for further analysis
    book_counts.to_csv("analysis/book_counts.csv", index=False)
    field_stats.to_csv("analysis/field_presence_stats.csv", index=False)
    entry_stats.to_csv("analysis/field_entry_stats.csv", index=False)
    
    print("\nResults saved to CSV files.")

if __name__ == "__main__":
    main()