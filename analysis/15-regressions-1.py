import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import json # To parse education string
import re # Potentially needed for HISCO cleaning
import logging
from typing import Dict, Any, Optional, List


# --- Load Data ---
# Assume 'df' is your DataFrame loaded from 'persons_data_with_networks.csv'
# Example: df = pd.read_csv("data/analysis/persons_data_with_networks.csv", ...) 
# Make sure to handle NA values appropriately on load if necessary
df = pd.read_csv("data/analysis/persons_data_with_networks.csv", na_filter=False) # Adjust path as needed


# helper functions


def safe_float_to_int(value: Any) -> Optional[int]:
    """Safely convert a value (potentially float string) to int."""
    if value is None:
        return None
    try:
        # Handle potential float strings like "21230.0"
        return int(float(value))
    except (ValueError, TypeError):
        return None
    

# --- 1. Data Preparation ---
logging.info("Preparing data for regression...")

# Convert existing boolean columns to 0/1
bool_cols = ['edu_technical', 'edu_business', 'edu_other_higher', 
             'career_has_overseas', 'career_has_us', 
             'worked_wl_before_1930'] # Note: worked_wl already boolean
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool).astype(int)
    else:
        logging.warning(f"Expected boolean column '{col}' not found.")

# --- Create 'father_social_class' ---
# Placeholder - requires specific HISCO mapping logic
def map_hisco_to_class(hisco_str):
    # Add your logic here:
    # 1. Clean hisco_str (remove decimals, handle errors)
    # 2. Convert to integer
    # 3. Compare against ranges for different classes
    # 4. Return a class label (e.g., "Elite", "WhiteCollar", "Skilled", "Farmer", "Unskilled", "Unknown")
    # Example (very basic, needs proper ranges):
    code = safe_float_to_int(hisco_str) # Use your existing helper
    if code is None: return "Unknown"
    if 10000 <= code < 20000: return "Elite" 
    if (20000 <= code < 40000) or (1000 <= code < 4000): return "WhiteCollar" # Broad example including engineers
    if 60000 <= code < 70000: return "Farmer"
    if 70000 <= code < 99000: return "Skilled"
    # Add more specific ranges based on HISCLASS or other schema
    return "Other/Unknown" # Default

logging.info("Creating father_social_class...")
df['father_social_class'] = df['father_hisco'].apply(map_hisco_to_class)
# Convert to category type for statsmodels
df['father_social_class'] = pd.Categorical(df['father_social_class'])
logging.info(f"Father social class distribution:\n{df['father_social_class'].value_counts(dropna=False)}")

# --- Create 'birth_cohort' category ---
logging.info("Creating birth_cohort category...")

# First, convert birth_decade to numeric, forcing errors to NaN
df['birth_decade'] = pd.to_numeric(df['birth_decade'], errors='coerce')

# Define bins and labels (adjust as needed)
bins = [0, 1880, 1900, 1920, np.inf]  # Bins: <1880, 1880-1899, 1900-1919, 1920+
labels = ['<1880', '1880-1899', '1900-1919', '1920+']

# Now create the bins
df['birth_cohort'] = pd.cut(df['birth_decade'], bins=bins, labels=labels, right=False)
logging.info(f"Birth cohort distribution:\n{df['birth_cohort'].value_counts(dropna=False)}")

# --- Create 'studied_at_[Institution]' dummies ---
# Placeholder - requires parsing JSON and checking names
logging.info("Creating studied_at_* dummies...")
# Define aliases for key institutions
kth_aliases = ["kungliga tekniska högskolan", "kth", "tekniska högskolan"] # Be careful with generic ones
chalmers_aliases = ["chalmers tekniska högskola", "chalmers", "cth", "chalmers tekniska institut", "cti", "chalmers tekniska läroanstalt"]
hhs_aliases = ["handelshögskolan i stockholm", "handelshögskolan stockholm", "hhs", "handelshögskolan, stockholm"]
foreign_strings = ["foreign study", "foreign university"] # Standardized names indicating foreign study

def check_study_location(edu_json_str, targets):
    try:
        edu_list = json.loads(edu_json_str)
        if not isinstance(edu_list, list): return 0
        for entry in edu_list:
            inst_std = entry.get("institution_standardized", "").lower()
            # Check if the standardized name matches any target alias or category
            if any(target in inst_std for target in targets):
                 return 1
    except: # Catch JSON parsing errors or other issues
        return 0
    return 0

df['studied_kth'] = df['education_standardized'].apply(lambda x: check_study_location(x, kth_aliases))
df['studied_chalmers'] = df['education_standardized'].apply(lambda x: check_study_location(x, chalmers_aliases))
df['studied_hhs'] = df['education_standardized'].apply(lambda x: check_study_location(x, hhs_aliases))
df['studied_foreign'] = df['education_standardized'].apply(lambda x: check_study_location(x, foreign_strings))

logging.info(f"Studied KTH: {df['studied_kth'].sum()}, Chalmers: {df['studied_chalmers'].sum()}, HHS: {df['studied_hhs'].sum()}, Foreign: {df['studied_foreign'].sum()}")

# --- Create 'birth_cohort' category ---
logging.info("Creating birth_cohort category...")
# Define bins and labels (adjust as needed)
bins = [0, 1880, 1900, 1920, np.inf] # Bins: <1880, 1880-1899, 1900-1919, 1920+
labels = ['<1880', '1880-1899', '1900-1919', '1920+']
df['birth_cohort'] = pd.cut(df['birth_decade'], bins=bins, labels=labels, right=False) # Ensure decade 1900 is in 1900-1919
logging.info(f"Birth cohort distribution:\n{df['birth_cohort'].value_counts(dropna=False)}")

# --- Prepare Dependent Variables ---
logging.info("Preparing dependent variables...")
# Model 1: Predicting Birth in WL
df_model1 = df.copy()
# Convert True/False/None to 1/0/NaN
df_model1['dep_var_birth_wl'] = df_model1['birth_parish_is_western_line'].map({True: 1, False: 0, None: np.nan})
# Drop rows where dependent variable is NaN
df_model1 = df_model1.dropna(subset=['dep_var_birth_wl'])
df_model1['dep_var_birth_wl'] = df_model1['dep_var_birth_wl'].astype(int)
logging.info(f"Model 1 using 'dep_var_birth_wl': {len(df_model1)} observations after dropping NA.")

# Model 2: Predicting Work in WL before 1930
df_model2 = df.copy()
# Variable is already 0/1 integer from boolean conversion
df_model2['dep_var_work_wl'] = df_model2['worked_wl_before_1930'] 
# No drop needed unless the column itself had NaNs initially (unlikely for bool)
logging.info(f"Model 2 using 'dep_var_work_wl': {len(df_model2)} observations.")

# --- Handle Missing Independent Variables (Example: Network Prop) ---
# Option: Drop rows with NaN in key predictors
cols_to_check_na = ['edu_network_wl_birth_prop', 'father_social_class', 'birth_cohort'] 
logging.info(f"Initial rows before dropping NAs in predictors: Model 1={len(df_model1)}, Model 2={len(df_model2)}")
df_model1 = df_model1.dropna(subset=cols_to_check_na)
df_model2 = df_model2.dropna(subset=cols_to_check_na)
logging.info(f"Final rows after dropping NAs in predictors: Model 1={len(df_model1)}, Model 2={len(df_model2)}")
# Option: Imputation (replace NaN with mean/median) - df_modelX[col].fillna(df_modelX[col].mean(), inplace=True)

# --- 2. Define and Run Models ---

# Define common independent variables (adjust list based on availability and relevance)
# Ensure variable names match columns created above
# Use C() for categorical variables
independent_vars = [
    'C(father_social_class)', 
    'C(birth_cohort)', 
    'edu_technical', 
    'edu_business', 
    'edu_other_higher', 
    'career_has_overseas', 
    'career_has_us', 
    'board_membership_count', 
    'studied_kth', # Add specific institution dummies
    'studied_chalmers',
    'studied_hhs',
    'studied_foreign',
    'edu_network_size', 
    'edu_network_wl_birth_prop' 
]
formula_rhs = " + ".join(independent_vars)

# --- Model 1: Predicting Birth in WL ---
if not df_model1.empty:
    logging.info("\n--- Running Probit Model 1: Predicting Birth in WL Parish ---")
    formula1 = f"dep_var_birth_wl ~ {formula_rhs}"
    logging.info(f"Formula: {formula1}")
    try:
        probit_model1 = smf.probit(formula=formula1, data=df_model1)
        results1 = probit_model1.fit()
        print(results1.summary())
    except Exception as e:
        logging.error(f"Error fitting Probit Model 1: {e}")
        logging.error("Check data types, missing values, and formula specification.")
        display_vars = [var.replace('C(', '').replace(')', '') if var.startswith('C(') else var 
                for var in independent_vars]
        logging.info(f"Data sample for Model 1:\n{df_model1[display_vars + ['dep_var_birth_wl']].head()}")

else:
    logging.warning("Skipping Probit Model 1: No data available after cleaning.")


# --- Model 2: Predicting Work in WL before 1930 ---
if not df_model2.empty:
    logging.info("\n--- Running Probit Model 2: Predicting Work in WL before 1930 ---")
    formula2 = f"dep_var_work_wl ~ {formula_rhs}"
    logging.info(f"Formula: {formula2}")
    try:
        probit_model2 = smf.probit(formula=formula2, data=df_model2)
        results2 = probit_model2.fit()
        print(results2.summary())
    except Exception as e:
        logging.error(f"Error fitting Probit Model 2: {e}")
        logging.error("Check data types, missing values, and formula specification.")
        logging.info(f"Data sample for Model 2:\n{df_model2[independent_vars + ['dep_var_work_wl']].head()}")

else:
    logging.warning("Skipping Probit Model 2: No data available after cleaning.")


logging.info("Regression analysis setup complete.")

# --- End of Script --- 
# Note: This script only sets up and runs the regressions. 
# Further analysis (marginal effects, cohort comparisons via sub-samples) requires additional code.