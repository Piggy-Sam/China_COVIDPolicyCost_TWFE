import pandas as pd
from linearmodels import PanelOLS
import numpy as np

# --- 1. Load, Reshape, and Merge Data (Same as before) ---
try:
    df_grp = pd.read_csv("China_COVID_measures_cost.xlsx - GRB (Real 2019 Billion RMB).csv")
    df_stringency = pd.read_csv("China_COVID_measures_cost.xlsx - OxCGRT Stringency Index.csv")
    df_cases = pd.read_csv("China_COVID_measures_cost.xlsx - COVID-19 New Confirmed PC Cases.csv")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    exit()

def reshape_to_panel(df, value_name):
    quarter_cols = [col for col in df.columns if 'Q' in col and col.startswith('20')]
    df_long = df.melt(id_vars=['GbProv', 'ProvEN'], value_vars=quarter_cols, var_name='Quarter', value_name=value_name)
    return df_long

panel_grp = reshape_to_panel(df_grp, 'GRP_real')
panel_stringency = reshape_to_panel(df_stringency, 'Stringency_Index')
panel_cases = reshape_to_panel(df_cases, 'Covid_Cases_per_mil')

df_panel = pd.merge(panel_grp, panel_stringency, on=['GbProv', 'ProvEN', 'Quarter'])
df_panel = pd.merge(df_panel, panel_cases, on=['GbProv', 'ProvEN', 'Quarter'])

# --- 2. Prepare Data and Create Lagged Variables ---

# Clean and prepare main variables
df_panel['Quarter'] = df_panel['Quarter'].str.strip()
df_panel = df_panel[df_panel['GRP_real'] > 0].copy()
df_panel['log_GRP'] = np.log(df_panel['GRP_real'])
df_panel['Covid_Cases'] = df_panel['Covid_Cases_per_mil'] / 1_000_000

# Convert Quarter to a sortable format first
df_panel['Time'] = df_panel['Quarter'].astype('category').cat.codes
df_panel = df_panel.sort_values(by=['GbProv', 'Time'])

# **NEW STEP: Create lagged variables**
# We group by province and shift the data by one period.
# This ensures we get the previous quarter's value for the same province.
df_panel['Stringency_Index_L1'] = df_panel.groupby('GbProv')['Stringency_Index'].shift(1)
df_panel['Covid_Cases_L1'] = df_panel.groupby('GbProv')['Covid_Cases'].shift(1)

# As discussed, fill the newly created NaN values for the first period (Q1 2020) with 0.
df_panel['Stringency_Index_L1'].fillna(0, inplace=True)
df_panel['Covid_Cases_L1'].fillna(0, inplace=True)

# Set the final index for the model
df_panel = df_panel.set_index(['GbProv', 'Time'])

# --- 3. Estimate the Model with Lagged Variables ---
dependent = df_panel['log_GRP']
# Add the new lagged variables to our list of exogenous regressors
exog_vars = ['Stringency_Index', 'Stringency_Index_L1', 'Covid_Cases', 'Covid_Cases_L1']
exog = df_panel[exog_vars]

model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
results_lagged = model.fit(cov_type='clustered', cluster_entity=True)

# --- 4. Print the Results ---
print("\n==============================================================================")
print("     TWFE Panel Regression Results with Lagged Independent Variables")
print("==============================================================================")
print(f"Dependent Variable: log(GRP)")
print(f"Number of Provinces: {df_panel.index.get_level_values('GbProv').nunique()}")
print(f"Total Observations: {results_lagged.nobs}")
print("------------------------------------------------------------------------------")
print(results_lagged)
print("==============================================================================")
print("\nInterpretation:")
try:
    beta_1 = results_lagged.params['Stringency_Index']
    beta_2 = results_lagged.params['Stringency_Index_L1']
    pval_1 = results_lagged.pvalues['Stringency_Index']
    pval_2 = results_lagged.pvalues['Stringency_Index_L1']

    print(f"\nContemporaneous Effect (t):")
    print(f"The coefficient for Stringency_Index in the current quarter is {beta_1:.5f} (p-value: {pval_1:.4f}).")
    print("\nLagged Effect (t-1):")
    print(f"The coefficient for Stringency_Index in the previous quarter is {beta_2:.5f} (p-value: {pval_2:.4f}).")

except KeyError:
    print("Could not find expected variables in the model results.")