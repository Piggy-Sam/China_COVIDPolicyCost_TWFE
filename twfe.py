import pandas as pd
from linearmodels import PanelOLS
import numpy as np

# --- 1. Load Data ---
try:
    df_grp = pd.read_csv("China_COVID_measures_cost.xlsx - GRB (Real 2019 Billion RMB).csv")
    df_stringency = pd.read_csv("China_COVID_measures_cost.xlsx - OxCGRT Stringency Index.csv")
    df_cases = pd.read_csv("China_COVID_measures_cost.xlsx - COVID-19 New Confirmed PC Cases.csv")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    exit()

# --- 2. Reshape Data ---
def reshape_to_panel(df, value_name):
    quarter_cols = [col for col in df.columns if 'Q' in col and col.startswith('20')]
    df_long = df.melt(id_vars=['GbProv', 'ProvEN'], value_vars=quarter_cols, var_name='Quarter', value_name=value_name)
    return df_long

panel_grp = reshape_to_panel(df_grp, 'GRP_real')
panel_stringency = reshape_to_panel(df_stringency, 'Stringency_Index')
panel_cases = reshape_to_panel(df_cases, 'Covid_Cases_per_mil')

# --- 3. Merge Data ---
df_panel = pd.merge(panel_grp, panel_stringency, on=['GbProv', 'ProvEN', 'Quarter'])
df_panel = pd.merge(df_panel, panel_cases, on=['GbProv', 'ProvEN', 'Quarter'])

# --- 4. Prepare Data for Regression ---

# Clean whitespace from quarter column FIRST
df_panel['Quarter'] = df_panel['Quarter'].str.strip()

# **NEW DEBUGGING STEP: Check for duplicate province-quarter entries**
duplicates = df_panel[df_panel.duplicated(subset=['GbProv', 'Quarter'], keep=False)]
if not duplicates.empty:
    print("!!! WARNING: Found duplicate entries for the same province and quarter !!!")
    print(duplicates)
    # Depending on the issue, you might want to stop here by uncommenting the next line
    # exit()
else:
    print("--- No duplicate province-quarter entries found. Proceeding. ---\n")


# Filter non-positive GRP values
df_panel = df_panel[df_panel['GRP_real'] > 0].copy()
df_panel['log_GRP'] = np.log(df_panel['GRP_real'])
df_panel['Covid_Cases'] = df_panel['Covid_Cases_per_mil'] / 1_000_000

# **FINAL FIX: Convert 'Quarter' to a simple numeric index**
# This is the most robust way to ensure the time index is recognized.
df_panel['Time'] = df_panel['Quarter'].astype('category').cat.codes
print("--- Converted 'Quarter' to numeric 'Time' column ---")
print(df_panel[['Quarter', 'Time']].drop_duplicates().sort_values('Time').head())
print("...")
print(df_panel[['Quarter', 'Time']].drop_duplicates().sort_values('Time').tail())
print("-------------------------------------------------")


# Set up the panel data structure using the new 'Time' column
df_panel = df_panel.set_index(['GbProv', 'Time'])


# --- 5. Estimate the Model ---
dependent = df_panel['log_GRP']
exog_vars = ['Stringency_Index', 'Covid_Cases']
exog = df_panel[exog_vars]

# This model specification now uses a numeric time index, which should resolve the error.
# We still include time_effects=True, which will treat each integer (0, 1, 2...) as a separate time period.
model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
results = model.fit(cov_type='clustered', cluster_entity=True)

# --- 6. Print the Results ---
print("\n==============================================================================")
print("       Two-Way Fixed Effects (TWFE) Panel Regression Results")
print("==============================================================================")
print(f"Dependent Variable: log(GRP)")
print(f"Number of Provinces: {df_panel.index.get_level_values('GbProv').nunique()}")
print(f"Total Observations: {results.nobs}")
print("------------------------------------------------------------------------------")
print(results)
print("==============================================================================")
print("\nInterpretation of the Key Coefficient (Stringency_Index):")
try:
    beta_1 = results.params['Stringency_Index']
    percent_change = (np.exp(beta_1) - 1) * 100
    print(f"A one-unit increase in the Stringency Index is associated with a {percent_change:.4f}% change in quarterly GRP, holding COVID cases and fixed effects constant.")
except KeyError:
    print("Could not find 'Stringency_Index' in the model results.")