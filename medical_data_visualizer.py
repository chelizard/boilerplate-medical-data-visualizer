import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("medical_examination.csv")

BMI = df['weight'] / (df['height'] / 100)**2
df['overweight'] = (BMI > 25).astype(int)

df[['cholesterol', 'gluc']] = (df[['cholesterol', 'gluc']] > 1).astype(int)

def draw_cat_plot(df):
    # Melt DataFrame for categorical plot
    df_cat = pd.melt(df, id_vars=["cardio"], 
                      value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat.rename(columns={0: 'total'}, inplace=True)
    
    sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    plt.show()

    # Get the figure for the output and store it in the fig variable.
    fig = plt.figure(figsize=(10, 10))
    # Save the figure and return it.
    fig.savefig('catplot.png')
    return fig

def draw_heat_map(df):
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(corr)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the correlation matrix using sns.heatmap.
    sns.heatmap(corr, mask=mask, center=0, vmin=-0.15, vmax=0.30, square=True, linewidths=.5, 
                annot=True, fmt='.1f',
                annot_kws={'size': 8},
                cbar_kws={'shrink': 0.25,
                          'ticks': [-0.08, 0.00, 0.08, 0.16, 0.24]},
                          ax=ax)
    plt.tight_layout()
    plt.show()

    # Save the figure and return it.
    fig.savefig('heatmap.png')
    return fig
    


    
# draw_cat_plot(df)
# print(df['overweight'].value_counts())
draw_heat_map(df)



