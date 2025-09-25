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

# --- 2. Load, Reshape, and Merge Data ---
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

# --- 3. Merge Urbanization Data ---
df_panel = pd.merge(df_panel, df_urban, on='ProvEN', how='left')
df_panel.dropna(subset=['Urbanization_Rate'], inplace=True)

# --- 4. Prepare Data, Create Lagged Variables and Interaction Terms ---
df_panel['Quarter'] = df_panel['Quarter'].str.strip()
df_panel = df_panel[df_panel['GRP_real'] > 0].copy()

# Sort values by province and time to ensure correct lagging
df_panel['Time'] = df_panel['Quarter'].astype('category').cat.codes
df_panel = df_panel.sort_values(by=['GbProv', 'Time'])

# Create lagged variables
df_panel['Stringency_Index_L1'] = df_panel.groupby('GbProv')['Stringency_Index'].shift(1)
df_panel['Covid_Cases_L1'] = df_panel.groupby('GbProv')['Covid_Cases_per_mil'].shift(1)

# Fill NaNs created by the lag with 0
df_panel.fillna(0, inplace=True)

# Create interaction terms for both current and lagged stringency
df_panel['Stringency_x_Urban'] = df_panel['Stringency_Index'] * df_panel['Urbanization_Rate']
df_panel['Stringency_L1_x_Urban'] = df_panel['Stringency_Index_L1'] * df_panel['Urbanization_Rate']

# Final data preparation
df_panel['log_GRP'] = np.log(df_panel['GRP_real'])
df_panel['Covid_Cases'] = df_panel['Covid_Cases_per_mil'] / 1_000_000
df_panel['Covid_Cases_L1'] = df_panel['Covid_Cases_L1'] / 1_000_000
df_panel = df_panel.set_index(['GbProv', 'Time'])

# --- 5. Estimate the Combined Model ---
dependent = df_panel['log_GRP']
exog_vars = [
    'Stringency_Index', 
    'Stringency_Index_L1', 
    'Stringency_x_Urban', 
    'Stringency_L1_x_Urban', 
    'Covid_Cases',
    'Covid_Cases_L1'
]
exog = df_panel[exog_vars]

model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
results_combined = model.fit(cov_type='clustered', cluster_entity=True)

# --- 6. Print the Results ---
print("\n==============================================================================")
print("     TWFE Results with Lagged Terms and Urbanization Interaction")
print("==============================================================================")
print(results_combined)
print("==============================================================================")