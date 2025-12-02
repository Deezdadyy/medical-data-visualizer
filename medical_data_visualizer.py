import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv("medical_examination.csv")

# 2. Add overweight column
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalize cholesterol and gluc (0 = good, 1 = bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # 5. Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and reformat the data to split it by 'cardio' and get counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the categorical plot with seaborn
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # 8. Get the figure for output (fig already set)
    return fig


def draw_heat_map():
    # 11. Clean the data
    df_heat = df.copy()

    # Remove incorrect blood pressure rows (diastolic > systolic)
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    # Remove height outliers (2.5th and 97.5th percentiles)
    h_low = df_heat['height'].quantile(0.025)
    h_high = df_heat['height'].quantile(0.975)
    df_heat = df_heat[(df_heat['height'] >= h_low) & (df_heat['height'] <= h_high)]

    # Remove weight outliers (2.5th and 97.5th percentiles)
    w_low = df_heat['weight'].quantile(0.025)
    w_high = df_heat['weight'].quantile(0.975)
    df_heat = df_heat[(df_heat['weight'] >= w_low) & (df_heat['weight'] <= w_high)]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Draw the heatmap with seaborn
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap='coolwarm',
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax,
        vmax=0.3,
        vmin=-0.16
    )

    # 16. Do not modify the next two lines (they return the figure)
    return fig
