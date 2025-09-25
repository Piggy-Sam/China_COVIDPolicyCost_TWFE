import pandas as pd
from linearmodels import PanelOLS
import numpy as np

# --- 1. Urbanization Data ---
urbanization_data = {
    'Shanghai': 89.46, 'Beijing': 87.83, 'Tianjin': 85.49, 'Guangdong': 75.42,
    'Jiangsu': 75.04, 'Zhejiang': 74.23, 'Liaoning': 73.51, 'Chongqing': 71.67,
    'Fujian': 71.04, 'Neimenggu': 69.58, 'Ningxia': 67.31, 'Heilongjiang': 67.11,
    'Shandong': 65.53, 'Hubei': 65.47, 'Shaanxi': 65.16, 'Shanxi': 64.97,
    'Jilin': 64.73, 'Jiangxi': 63.13, 'Qinghai': 62.80, 'Hebei': 62.77,
    'Hainan': 62.46, 'Anhui': 61.51, 'Hunan': 61.16, 'Sichuan': 59.49,
    'Xinjiang': 59.24, 'Henan': 58.08, 'Guangxi': 56.78, 'Guizhou': 55.94,
    'Gansu': 55.49, 'Yunnan': 52.92, 'Xizang': 38.88
}
df_urban = pd.DataFrame(list(urbanization_data.items()), columns=['ProvEN', 'Urbanization_Rate'])


# --- 2. Load, Reshape, and Merge Panel Data ---
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

# --- 3. Merge Urbanization Data and Create Interaction Term ---
df_panel = pd.merge(df_panel, df_urban, on='ProvEN', how='left')
df_panel.dropna(subset=['Urbanization_Rate'], inplace=True)
df_panel['Stringency_x_Urban'] = df_panel['Stringency_Index'] * df_panel['Urbanization_Rate']

# --- 4. Prepare Data for Regression ---
df_panel['Quarter'] = df_panel['Quarter'].str.strip()
df_panel = df_panel[df_panel['GRP_real'] > 0].copy()
df_panel['log_GRP'] = np.log(df_panel['GRP_real'])
df_panel['Covid_Cases'] = df_panel['Covid_Cases_per_mil'] / 1_000_000
df_panel['Time'] = df_panel['Quarter'].astype('category').cat.codes
df_panel = df_panel.set_index(['GbProv', 'Time'])

# --- 5. Estimate the Interaction Model ---
dependent = df_panel['log_GRP']
# **FIX**: The time-invariant 'Urbanization_Rate' is dropped because its effect
# is absorbed by the province-level fixed effects. We keep the interaction term.
exog_vars = ['Stringency_Index', 'Stringency_x_Urban', 'Covid_Cases']
exog = df_panel[exog_vars]

model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
results_interaction = model.fit(cov_type='clustered', cluster_entity=True)


# --- 6. Print the Results ---
print("\n==============================================================================")
print("     TWFE Panel Regression Results with Urbanization Interaction Term")
print("==============================================================================")
print(results_interaction)
print("==============================================================================")
print("\nInterpretation of Key Coefficients:")
try:
    beta_1 = results_interaction.params['Stringency_Index']
    beta_3 = results_interaction.params['Stringency_x_Urban']
    pval_1 = results_interaction.pvalues['Stringency_Index']
    pval_3 = results_interaction.pvalues['Stringency_x_Urban']

    print(f"\nBaseline Effect (Stringency_Index): {beta_1:.6f} (p-value: {pval_1:.4f})")
    print(f"Interaction Effect (Stringency_x_Urban): {beta_3:.6f} (p-value: {pval_3:.4f})")
    
    if pval_3 < 0.1:
         print("\nThe interaction term is statistically significant, supporting the hypothesis.")
         urban_low = 55.49 # Gansu's Rate
         urban_high = 89.46 # Shanghai's Rate
         effect_low = (np.exp(beta_1 + beta_3 * urban_low) - 1) * 100
         effect_high = (np.exp(beta_1 + beta_3 * urban_high) - 1) * 100
         print(f"\nExample Total Effect for a 1-unit Stringency increase:")
         print(f" -> In a province like Gansu ({urban_low}% urbanization): {effect_low:.4f}% change in GRP.")
         print(f" -> In a province like Shanghai ({urban_high}% urbanization): {effect_high:.4f}% change in GRP.")
    else:
        print("\nThe interaction term is not statistically significant at conventional levels.")

except KeyError:
    print("Could not find expected variables in the model results.")