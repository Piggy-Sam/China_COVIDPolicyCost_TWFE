import pandas as pd
import numpy as np

# --- 1. Define the Core Parameter from our Regression ---
# This is the statistically significant coefficient from Model 2.
BETA_1_STRINGENCY = -0.00075

# --- 2. Load Necessary Data ---
try:
    # We need the actual observed Real GRP data
    df_grp_real = pd.read_csv("China_COVID_measures_cost.xlsx - GRB (Real 2019 Billion RMB).csv")
    # We need the Stringency Index data for each province-quarter
    df_stringency = pd.read_csv("China_COVID_measures_cost.xlsx - OxCGRT Stringency Index.csv")
except FileNotFoundError:
    print("Error: Ensure data files are in the same directory.")
    exit()

# --- 3. Reshape Data to Long (Panel) Format ---
def reshape_to_panel(df, value_name):
    quarter_cols = [col for col in df.columns if 'Q' in col and col.startswith('20')]
    df_long = df.melt(
        id_vars=['GbProv', 'ProvEN'],
        value_vars=quarter_cols,
        var_name='Quarter',
        value_name=value_name
    )
    return df_long

panel_grp = reshape_to_panel(df_grp_real, 'GRP_real_actual')
panel_stringency = reshape_to_panel(df_stringency, 'Stringency_Index')

# --- 4. Merge into a Single DataFrame ---
df_analysis = pd.merge(panel_grp, panel_stringency, on=['GbProv', 'ProvEN', 'Quarter'])

# --- 5. Calculate Counterfactual GRP and Policy Cost ---
# Apply the formula to each row (each province-quarter observation)
df_analysis['GRP_counterfactual'] = df_analysis['GRP_real_actual'] / np.exp(BETA_1_STRINGENCY * df_analysis['Stringency_Index'])

# The cost is the difference between the 'no-policy' scenario and what actually happened
df_analysis['Policy_Cost'] = df_analysis['GRP_counterfactual'] - df_analysis['GRP_real_actual']

# --- 6. Aggregate and Present the Final Result ---
# Sum the costs across all observations
total_cost_billion_rmb = df_analysis['Policy_Cost'].sum()

# Convert to a more readable format (Trillion RMB)
total_cost_trillion_rmb = total_cost_billion_rmb / 1000

# --- Print a clear, publication-ready summary ---
print("==============================================================================")
print("   Estimated Total Economic Cost of Zero-COVID Policies (2020-2022)")
print("==============================================================================")
print(f"Based on the estimated coefficient (β1) of: {BETA_1_STRINGENCY}")
print(f"Total calculated policy-attributable GRP loss (Billion 2019 RMB): {total_cost_billion_rmb:,.2f}")
print(f"Total calculated policy-attributable GRP loss (Trillion 2019 RMB): {total_cost_trillion_rmb:,.2f}")
print("==============================================================================")