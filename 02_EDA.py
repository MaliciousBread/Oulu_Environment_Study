# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

plt.rcParams["figure.figsize"] = (12, 6)

print("Bibliothèques importées avec succès.")
# %%

# %%
print("Chargement et fusion des données...")
df_sensor = pd.read_csv('Projet_Data/application.csv')
df_network = pd.read_csv('Projet_Data/lora.csv')

df_sensor['time'] = pd.to_datetime(df_sensor['time'], unit='ms')
df_network['time'] = pd.to_datetime(df_network['time'], unit='ms')

df = pd.merge(df_sensor, df_network, on=['time', 'deveui'], how='inner')
df = df.sort_values("time").reset_index(drop=True)

print(f"Dimensions du dataset : {df.shape}")
# %%

# %%
env_cols = ['temperature', 'humidity', 'light', 'co2', 'moisture', 'pressure']
network_cols = ['rssi', 'lsnr', 'fcnt', 'seqn']

cols_for_corr = [c for c in env_cols + network_cols if c in df.columns]
corr = df[cols_for_corr].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Matrice de corrélation - Variables IoT")
plt.show()
# %%

# %%
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["year"] = df["time"].dt.year
df["day_of_week"] = df["time"].dt.day_name()
df["week"] = df["time"].dt.isocalendar().week.astype(int)
# %%

# %%
target_cols = ['temperature', 'humidity', 'co2']
targets_present = [c for c in target_cols if c in df.columns]

fig, axes = plt.subplots(1, len(targets_present), figsize=(18, 5))

for i, col in enumerate(targets_present):
    hourly_mean = df.groupby("hour")[col].mean()
    axes[i].plot(hourly_mean.index, hourly_mean.values, marker="o", color='darkorange')
    axes[i].set_title(f"Moyenne horaire - {col.capitalize()}")
    axes[i].set_xlabel("Heure")
    axes[i].set_ylabel(col)
    axes[i].grid(True)

plt.tight_layout()
plt.show()
# %%

# %%
target_cols = ['temperature', 'humidity', 'co2']
targets_present = [c for c in target_cols if c in df.columns]

fig, axes = plt.subplots(1, len(targets_present), figsize=(18, 5))

for i, col in enumerate(targets_present):
    if col == 'temperature':
        valid_data = df.loc[(df[col] >= -20) & (df[col] <= 60), col]
    elif col == 'co2':
        valid_data = df.loc[df[col] <= 10000, col]
    elif col == 'humidity':
        valid_data = df.loc[(df[col] >= 0) & (df[col] <= 100), col]
    else:
        valid_data = df[col]
        
    sns.boxplot(x=valid_data, ax=axes[i], color="lightblue")
    axes[i].set_title(f"Distribution - {col.capitalize()} (Échelle fixée)")

plt.tight_layout()
plt.show()
# %%

# %%
targets = ['target_temp', 'target_humidity', 'target_co2']

for col, target_name in zip(targets_present, targets):
    df[target_name] = df[col].shift(-1)

print("Aperçu des 3 cibles décalées pour le prochain pas de temps :")
display(df[["time"] + targets_present + targets].head(10))
# %%
