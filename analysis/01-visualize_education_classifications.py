import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def visualize_education_by_decade():
    # Load the analysis data
    with open('analysis/engineer_education_analysis.json', 'r') as file:
        data = json.load(file)
    
    # Extract education by decade data
    by_decade = data['engineers_by_decade']
    
    # Convert to DataFrame for easier plotting
    decades = []
    education_levels = []
    counts = []
    
    degree_order = ["Schooling", "Bachelor's", "Master's", "Doctorate"]
    
    for decade, levels in by_decade.items():
        for level, count in levels.items():
            decades.append(decade)
            education_levels.append(level)
            counts.append(count)
    
    df = pd.DataFrame({
        'Decade': decades,
        'Education Level': education_levels,
        'Count': counts
    })
    
    # Ensure decades are sorted chronologically
    decade_order = sorted(df['Decade'].unique(), key=lambda x: int(x[:-1]))
    
    # Create a pivot table for stacked bar chart
    pivot_df = df.pivot_table(
        index='Decade', 
        columns='Education Level',
        values='Count',
        aggfunc='sum'
    ).fillna(0)
    
    # Reorder columns by education level
    pivot_df = pivot_df.reindex(columns=degree_order)
    
    # Reorder rows by decade
    pivot_df = pivot_df.reindex(decade_order)
    
    # Plot stacked bar chart
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('Highest Education Level of Engineers by Decade of Birth', fontsize=16)
    plt.xlabel('Decade of Birth', fontsize=14)
    plt.ylabel('Number of Engineers', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Education Level', title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig('analysis/engineer_education_by_decade.png', dpi=300)
    print("Saved visualization to engineer_education_by_decade.png")
    
    # Calculate percentages for each decade
    percentage_df = pivot_df.copy()
    for decade in percentage_df.index:
        total = percentage_df.loc[decade].sum()
        if total > 0:  # Avoid division by zero
            percentage_df.loc[decade] = (percentage_df.loc[decade] / total) * 100
    
    # Plot percentage stacked bar chart
    plt.figure(figsize=(12, 8))
    percentage_df.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('Proportion of Education Levels Among Engineers by Decade of Birth', fontsize=16)
    plt.xlabel('Decade of Birth', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Education Level', title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig('analysis/engineer_education_percentage_by_decade.png', dpi=300)
    print("Saved percentage visualization to engineer_education_percentage_by_decade.png")
    
    # Analyze educational institutions
    analyze_institutions(data['unique_institutions'])

def analyze_institutions(institutions_data):
    # Create a DataFrame for institutions
    df = pd.DataFrame(institutions_data)
    
    # Only include top institutions (those with more than 1% of the total)
    total_count = df['count'].sum()
    threshold = total_count * 0.01
    
    top_institutions = df[df['count'] >= threshold].copy()
    other_count = df[df['count'] < threshold]['count'].sum()
    
    # Add an "Other" category
    if other_count > 0:
        top_institutions = pd.concat([
            top_institutions,
            pd.DataFrame([{'name': 'Other Institutions', 'count': other_count}])
        ], ignore_index=True)
    
    # Sort by count
    top_institutions = top_institutions.sort_values('count', ascending=False)
    
    # Plot bar chart
    plt.figure(figsize=(14, 8))
    sns.barplot(x='count', y='name', data=top_institutions, palette='viridis')
    
    plt.title('Distribution of Engineers by Educational Institution', fontsize=16)
    plt.xlabel('Number of Engineers', fontsize=14)
    plt.ylabel('Institution', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('analysis/engineer_institutions.png', dpi=300)
    print("Saved institutions visualization to engineer_institutions.png")
    
    # Create a classification template for institutions
    create_institution_classification_template(institutions_data)

def create_institution_classification_template(institutions_data):
    # Create a template for manual classification
    classification = {}
    
    for inst in institutions_data:
        name = inst['name']
        # Initial guess at standardized name - remove common prefixes
        std_name = name
        for prefix in ['Kungliga ', 'Royal ', 'The ']:
            if std_name.startswith(prefix):
                std_name = std_name[len(prefix):]
        
        classification[name] = {
            'standardized_name': std_name,
            'type': 'university',  # Default, can be changed manually
            'count': inst['count']
        }
    
    # Save the template
    with open('analysis/institution_classification_template.json', 'w') as file:
        json.dump(classification, file, indent=2, ensure_ascii=False)
    
    print("Created institution classification template: institution_classification_template.json")

if __name__ == "__main__":
    visualize_education_by_decade()