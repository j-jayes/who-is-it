import os
import json
import pandas as pd
import geopandas as gpd # Required external library
import statsmodels.formula.api as smf # Required external library
from glob import glob
import logging
from collections import defaultdict 
from typing import Dict, List, Any, Optional, Tuple, Set
import re
import numpy as np # Required external library

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_DIR: str = "data/enriched_biographies"
OUTPUT_DIR: str = "data/analysis"
PARISH_SHAPEFILE_PATH: str = "data/parishes/parish_map_1920.shp"
INSTITUTION_MAPPING_PATH: str = os.path.join(OUTPUT_DIR, "institution_mapping.json") 
FINAL_CSV_OUTPUT_PATH: str = os.path.join(OUTPUT_DIR, 'persons_data_final_for_regression.csv') 

os.makedirs(OUTPUT_DIR, exist_ok=True)

ENGINEER_HISCO_RANGES: List[Tuple[int, int]] = [(1000, 3999)] 
DIRECTOR_HISCO_RANGES: List[Tuple[int, int]] = [(21000, 21900)] 
RELEVANT_HISCO_RANGES: List[Tuple[int, int]] = ENGINEER_HISCO_RANGES + DIRECTOR_HISCO_RANGES
CAREER_YEAR_THRESHOLD: int = 1930
SHP_PARISH_CODE_COL: str = 'prsh_cd'; SHP_PARISH_NAME_COL: str = 'prsh_nm'; SHP_WESTERN_LINE_COL: str = 'wstrn_l'; SHP_GEOMETRY_COL: str = 'geometry'
COHORT_YEAR_WINDOW: int = 4 



def main() -> int:
    """Main execution function."""
    logging.info(f"Starting data extraction, mapping, and standardization...")

    # --- 1. Load Data ---
    final_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'persons_data_final_for_regression.csv'), encoding='utf-8-sig')

    logging.info("Final cleaning and preparation for regression models...")
    bool_cols_to_int = ['edu_technical', 'edu_business', 'edu_other_higher', 'career_has_overseas', 'career_has_us', 'worked_wl_before_1930', 'worked_wl_after_1930', 'studied_kth', 'studied_chalmers', 'studied_hhs', 'studied_foreign']
    for col in bool_cols_to_int:
        if col in final_df.columns: final_df[col] = final_df[col].astype(bool).astype(int)
    final_df['dep_var_birth_wl'] = final_df['birth_parish_is_western_line'].map({True: 1, False: 0}); df_model1 = final_df.dropna(subset=['dep_var_birth_wl']).copy(); df_model1['dep_var_birth_wl'] = df_model1['dep_var_birth_wl'].astype(int)
    final_df['dep_var_work_wl'] = final_df['worked_wl_before_1930']; df_model2 = final_df.copy() 
    

    # Define independent variables & Handle remaining NaNs
    independent_vars = ['C(father_hisco_major_group_label)', 'C(birth_cohort)', 'edu_technical', 'edu_business', 'edu_other_higher', 'career_has_overseas', 'career_has_us', 'board_membership_count', 'studied_kth', 'studied_chalmers', 'studied_hhs', 'studied_foreign', 'edu_network_size', 'edu_network_wl_birth_prop' 
                        ]
    predictors_for_na_check = [col for col in independent_vars if not col.startswith('C(')] # Check continuous/dummy vars for NA
    predictors_for_na_check.extend(['father_hisco_major_group_label', 'birth_cohort']) # Also check categoricals before dummy creation by statsmodels
    
    # **** TROUBLESHOOTING: Impute network proportion with median ****
    median_net_prop = df_model1['edu_network_wl_birth_prop'].median() # Use median from model 1's data
    logging.info(f"Imputing NaN in 'edu_network_wl_birth_prop' with median: {median_net_prop:.4f}")
    # **** FIX: Address FutureWarning ****
    df_model1['edu_network_wl_birth_prop'] = df_model1['edu_network_wl_birth_prop'].fillna(median_net_prop)
    df_model2['edu_network_wl_birth_prop'] = df_model2['edu_network_wl_birth_prop'].fillna(median_net_prop)

    # Now drop rows only if other essential predictors are missing (less likely now)
    # Check if 'father_hisco_major_group_label' or 'birth_cohort' have actual NaNs (not 'Unknown')
    essential_predictors = ['father_hisco_major_group_label', 'birth_cohort'] # Add others if critical
    initial_rows1 = len(df_model1); initial_rows2 = len(df_model2)
    df_model1 = df_model1.dropna(subset=essential_predictors) 
    df_model2 = df_model2.dropna(subset=essential_predictors)
    logging.info(f"Rows for Model 1 after essential NA drop: {len(df_model1)} (out of {initial_rows1})")
    logging.info(f"Rows for Model 2 after essential NA drop: {len(df_model2)} (out of {initial_rows2})")
    logging.info(f"Missing values count for edu_network_wl_birth_prop after imputation: Model 1={df_model1['edu_network_wl_birth_prop'].isna().sum()}, Model 2={df_model2['edu_network_wl_birth_prop'].isna().sum()}")

    # --- 8. Define and Run Probit Models ---
    formula_rhs = " + ".join(independent_vars)
    
    # --- Model 1: Predicting Birth in WL Parish ---
    if not df_model1.empty:
        logging.info("\n--- Running Probit Model 1: Predicting Birth in WL Parish ---")
        logging.info(f"Model 1 Dep Var Distribution:\n{df_model1['dep_var_birth_wl'].value_counts(normalize=True)}")

        if df_model1['dep_var_birth_wl'].nunique() < 2:
            logging.error("Model 1 dependent variable has only one outcome. Cannot fit Probit model.")
        else:
            # **** DEBUGGING MODEL 1: Start simple and add variables back ****
            try:
                # Start with a minimal set of predictors
                independent_vars_m1 = [
                    'C(birth_cohort)', 
                    'C(father_hisco_major_group_label)',
                    # --- Uncomment next group ---
                    'edu_technical', 
                    'edu_business', 
                    'edu_other_higher', 
                    # --- Uncomment next group ---
                    'career_has_overseas', 
                    'career_has_us', 
                    'board_membership_count', 
                    # --- Uncomment next group ---
                    'studied_kth', 
                    'studied_chalmers',
                    'studied_hhs',
                    'studied_foreign',
                    # --- Uncomment next group ---
                    'edu_network_size', 
                    'edu_network_wl_birth_prop' 
                ]
                formula_rhs_m1 = " + ".join(independent_vars_m1)
                formula1 = f"dep_var_birth_wl ~ {formula_rhs_m1}"
                logging.info(f"Running Model 1 with Formula: {formula1}")

                probit_model1 = smf.probit(formula=formula1, data=df_model1)
                # You might try 'bfgs' if default Newton-Raphson fails even on simple models
                # results1 = probit_model1.fit(maxiter=200, method='bfgs') 
                results1 = probit_model1.fit(maxiter=200) 

                print("\n--- Probit Model 1 Summary ---")
                print(results1.summary())
                if np.isnan(results1.llf): logging.warning("Model 1 resulted in NaN log-likelihood.")
                if not results1.mle_retvals['converged']: logging.warning("Model 1 failed to converge.")

            except np.linalg.LinAlgError as e:
                logging.error(f"LinAlgError (Singular matrix) fitting Probit Model 1 with current formula: {formula1}")
                logging.error("This likely indicates perfect multicollinearity or separation.")
                logging.error("Try removing more variables or examining correlations/crosstabs for the included predictors.")
            except Exception as e: 
                logging.error(f"Other error fitting Probit Model 1: {e}", exc_info=True)
    else: 
        logging.warning("Skipping Probit Model 1: No data available after cleaning.")

    # --- Model 2 Section (Keep as is, assuming it worked) ---
    # ... (rest of Model 2 code) ...

    # --- Model 2 ---
    # **** TROUBLESHOOTING: Simplify model if quasi-separation suspected ****
    formula_rhs_model2 = formula_rhs # Start with full formula
    # Example: Comment out father's class if it caused issues previously
    # independent_vars_m2 = [v for v in independent_vars if 'father_hisco' not in v]
    # formula_rhs_model2 = " + ".join(independent_vars_m2)
    # logging.warning("Running Model 2 with simplified formula (excluding father's HISCO group)")

    if not df_model2.empty:
        logging.info("\n--- Running Probit Model 2: Predicting Work in WL before 1930 ---")
        formula2 = f"dep_var_work_wl ~ {formula_rhs_model2}"
        logging.info(f"Formula: {formula2}")
        # **** TROUBLESHOOTING: Check dependent variable distribution ****
        logging.info(f"Model 2 Dep Var Distribution:\n{df_model2['dep_var_work_wl'].value_counts(normalize=True)}")
        if df_model2['dep_var_work_wl'].nunique() < 2:
            logging.error("Model 2 dependent variable has only one outcome. Cannot fit Probit model.")
        else:
            try:
                probit_model2 = smf.probit(formula=formula2, data=df_model2)
                results2 = probit_model2.fit(maxiter=200) # Increased maxiter
                print("\n--- Probit Model 2 Summary ---"); print(results2.summary())
                # Check convergence warning
                if not results2.mle_retvals['converged']:
                     logging.warning("Model 2 failed to converge. Results may be unreliable. Check for quasi-separation.")
                     # Optionally, try simplifying the model further or using a different solver as for Model 1.
            except Exception as e: logging.error(f"Error fitting Probit Model 2: {e}", exc_info=True)
    else: logging.warning("Skipping Probit Model 2: No data.")

    # # --- 9. Save Final Processed DataFrame ---
    # try:
    #     final_df.to_csv(FINAL_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig') 
    #     logging.info(f"Saved final processed data ({len(final_df)} rows) to: {FINAL_CSV_OUTPUT_PATH}")
    # except Exception as e: logging.error(f"Failed to save final CSV: {e}"); return 1

    logging.info("Processing finished successfully.")
    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    try: import geopandas; logging.info(f"Geopandas version: {geopandas.__version__}")
    except ImportError: logging.error("Required module 'geopandas' not found."); exit(1) 
    try: import statsmodels; logging.info(f"Statsmodels version: {statsmodels.__version__}")
    except ImportError: logging.error("Required module 'statsmodels' not found."); exit(1) 
    main()