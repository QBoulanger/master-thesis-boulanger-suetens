import pandas as pd

# Charger le fichier CSV
df = pd.read_csv(f'wumpus_data/csv_files/data_all.csv')

# Calculer la moyenne de la performance pour chaque nombre d'épisode pour chaque ID
result_df = df.groupby(['id', 'number_of_episodes', "legend"])['perf'].mean().reset_index()

dataframes = {}
for episode in result_df['number_of_episodes'].unique():
    episode_df = result_df[result_df['number_of_episodes'] == episode]
    episode_df_sorted = episode_df.sort_values(by='perf')
    dataframes[episode] = episode_df_sorted

# Affichage des DataFrames pour chaque nombre d'épisodes trié par perf dans l'ordre croissant
for episode, dataframe in dataframes.items():
    print(f"DataFrame pour {episode} épisodes trié par perf dans l'ordre croissant:")
    print(dataframe)
    dataframe.to_csv(f'{episode}.csv', index=False)
    print("\n")