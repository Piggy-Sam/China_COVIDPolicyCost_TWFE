import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the real GRP data
try:
    df_grp_raw = pd.read_csv("China_COVID_measures_cost.xlsx - GRB (Real 2019 Billion RMB).csv")
except FileNotFoundError:
    print("Could not find the GRP data file.")
    exit()

# Reshape to long format for plotting
quarter_cols = [col for col in df_grp_raw.columns if 'Q' in col and col.startswith('20')]
panel_grp = df_grp_raw.melt(
    id_vars=['ProvEN'],
    value_vars=quarter_cols,
    var_name='Quarter',
    value_name='GRP_real'
)

# Select a few representative provinces to avoid a cluttered plot
provinces_to_plot = ['Beijing', 'Guangdong', 'Hubei', 'Xinjiang']
df_plot = panel_grp[panel_grp['ProvEN'].isin(provinces_to_plot)]

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

sns.lineplot(data=df_plot, x='Quarter', y='GRP_real', hue='ProvEN', marker='o', ax=ax)

ax.set_title('Quarterly Real GRP for Select Provinces (2020-2022)', fontsize=16, fontweight='bold')
ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Real GRP (2019 Billion RMB)', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Province')
plt.tight_layout()
plt.show()