import os
import nfl_data_py as nfl
import polars as pl
import numpy as np
from ebbpy import add_ebb_estimate



def load_local_pbp(filename='pbp.csv', seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]):
    # Check if 'pbp.csv' exists in the root directory
    if os.path.exists(filename):
        print("Loading PBP data locally from " + filename +  "...")
        # Load PBP data from the CSV file
        pbp_raw = pl.read_csv(filename)
    else:
        print("Downloading PBP data...")
        # Download PBP data for the specified years
        pbp_raw = nfl.import_pbp_data(seasons)
        
        # Save the PBP data as 'pbp_raw_[min_year]_[max_year].csv'
        min_year = min(seasons)
        max_year = max(seasons)
        #pbp_filename = f'pbp_raw_{min_year}_{max_year}.csv'
        
        # Convert to pandas DataFrame to save as CSV
        #pbp_raw.to_csv(pbp_filename, index=False)
        # print(f"PBP data saved as '{pbp_filename}'.")
        
        # Optionally, you can also save a copy as filename for future use
        pbp_raw.to_csv(filename, index=False)
        print(f"PBP data saved as '{filename}'.")
        
        # Convert to Polars DataFrame for processing
        pbp_raw = pl.from_pandas(pbp_raw)

    
    # Ensure 'pbp' is a Polars DataFrame
    if not isinstance(pbp_raw, pl.DataFrame):
        pbp_raw = pl.from_pandas(pbp_raw)

    return pbp_raw


def generate_primary_passers(pbp):
    # Step 1: Filter the data
    passers = (
        pbp.filter(
            (pl.col('score_differential').is_not_null()) &
            (pl.col('season_type') == "REG") &
            (pl.col('posteam').is_not_null()) & (pl.col('posteam') != "") &
            (pl.col('qb_dropback') == 1)
        )
    )

    # Step 2: Group by and count dropbacks
    passers = passers.group_by(
        ['game_id', 'home_team', 'away_team', 'posteam', 'season', 'passer_player_name', 'passer_player_id']
    ).agg(
        dropbacks=pl.len()
    )

    # Step 3: Rank dropbacks within each group
    passers = passers.with_columns(
        dropbacks_ct_rank=pl.col('dropbacks')
        .rank('ordinal', descending=True)
        .over(['game_id', 'posteam', 'season'])
    )

    # Step 4: Filter to primary passers
    passers = passers.filter(pl.col('dropbacks_ct_rank') == 1)

    # Step 5: Rename columns
    passers = passers.rename({
        'passer_player_name': 'posteam_primary_passer',
        'passer_player_id': 'posteam_passer_id',
        'dropbacks': 'posteam_primary_passer_dropbacks'
    })

    # Step 6: Aggregate to get home and away primary passers
    passers =  passers.group_by(['game_id', 'season']).agg([
    pl.first('home_team'),
    pl.first('away_team'),
    pl.col('posteam_passer_id')
    .filter(pl.col('posteam') == pl.col('home_team'))
    .first()
    .alias('home_primary_passer_id'),
    pl.col('posteam_primary_passer_dropbacks')
    .filter(pl.col('posteam') == pl.col('home_team'))
    .first()
    .alias('home_primary_passer_dropbacks'),
    pl.col('posteam_passer_id')
    .filter(pl.col('posteam') == pl.col('away_team'))
    .first()
    .alias('away_primary_passer_id'),
    pl.col('posteam_primary_passer_dropbacks')
    .filter(pl.col('posteam') == pl.col('away_team'))
    .first()
    .alias('away_primary_passer_dropbacks'),
    ])

    # Step 7: Merge back to the original PBP DataFrame
    pbp = pbp.join(passers, on=['game_id', 'season'], how='left')

    # Step 8: Create 'posteam_primary_passer_id' and 'posteam_primary_passer_dropbacks' columns
    pbp = pbp.with_columns([
        pl.when(pl.col('posteam') == pl.col('home_team'))
        .then(pl.col('home_primary_passer_id'))
        .when(pl.col('posteam') == pl.col('away_team'))
        .then(pl.col('away_primary_passer_id'))
        .otherwise(None)
        .alias('posteam_primary_passer_id'),
        pl.when(pl.col('posteam') == pl.col('home_team'))
        .then(pl.col('home_primary_passer_dropbacks'))
        .when(pl.col('posteam') == pl.col('away_team'))
        .then(pl.col('away_primary_passer_dropbacks'))
        .otherwise(None)
        .alias('posteam_primary_passer_dropbacks')
    ])

    return pbp




def calculate_tdsr(pbp_df, min_plays=15, rosters_df=None, scale_metrics=True):
    pbp_filtered = pbp_df.filter(pl.col('passer_player_id').is_not_null())

    # Step 3: Create 'qb' column
    pbp_filtered = pbp_filtered.with_columns(
        pl.col('passer_player_id').alias('qb'),
        pl.when(pl.col('fixed_drive_result') == 'Touchdown')
        .then(1.0)
        .otherwise(0.0)
        .alias('drive_ended_with_touchdown')
    )

    # Step 4: Determine the drive column
    drive_col = 'fixed_drive' if 'fixed_drive' in pbp_filtered.columns else 'drive'

    # Step 5: Group by drives and quarterbacks
    group_cols = ['game_id', 'season', 'posteam', drive_col, 'qb']

    pbp_grouped = (
        pbp_filtered
        .group_by(group_cols)
        .agg([
            pl.len().alias('plays'),
            pl.col('epa').sum().alias('epa'),
            pl.col('cpoe').mean().alias('cpoe'),
            pl.col('drive_ended_with_touchdown').max().alias('drive_success')
        ])
    )

    # Handle missing values and cap 'cpoe' at 8.0
    pbp_grouped = pbp_grouped.filter(
        pl.col('drive_success').is_not_null()
    )
    pbp_grouped = pbp_grouped.with_columns(
        pl.col('cpoe').clip(upper_bound=8.0)
    )

    # Step 6: Group by quarterback and season
    qb_season_grouped = (
        pbp_grouped
        .group_by(['qb', 'season'])
        .agg([
            pl.col('plays').sum(),
            pl.col('epa').mean(),
            pl.col('cpoe').mean(),
            pl.col('drive_success').sum().cast(pl.Int32),
            pl.len().alias('drives')
        ])
    )

    # Remove invalid data
    qb_season_grouped = qb_season_grouped.filter(
        (pl.col('plays') > min_plays)
    )

    # Step 7: Ensure prior_subset only includes valid data
    prior_subset = (
        (qb_season_grouped['plays'] > min_plays) &
        (qb_season_grouped['drive_success'].is_not_null()) &
        (qb_season_grouped['drive_success'] <= qb_season_grouped['drives'])
    )

    # Step 8: Apply add_ebb_estimate to calculate TDSR
    result_pl = add_ebb_estimate(
        qb_season_grouped,
        x='drive_success',
        n='drives',
        prior_subset=prior_subset
    )

    # Step 9: Select and rename columns
    result_pl = result_pl.select(['qb', '.fitted', 'epa', 'cpoe', 'plays', 'drive_success', 'drives', 'season'])
    result_pl = result_pl.with_columns(
        pl.col('.fitted').alias('tdsr')
    ).drop('.fitted')
    results_pl = result_pl.sort('tdsr', descending=True)

    # Step 10: Import roster data and merge
    if rosters_df is None:
        rosters_raw = nfl.__import_rosters(release="seasonal", years=seasons)
        rosters_df = pl.from_pandas(rosters_raw)
    else:
        rosters_df = pl.DataFrame(rosters_df)

    rosters_selected = rosters_df.select(['player_name', 'player_id', 'team', 'season'])
    rosters = rosters_selected.rename({'player_name': 'player', 'player_id': 'qb'})
    rosters = rosters.with_columns(pl.col('season').cast(pl.Int64))

    # Merge with the result data
    final_result_pl = result_pl.join(rosters, on=['qb', 'season'], how='left')

    # Step 11: Filter for the most recent season
    max_season = max(seasons)

    final_result_pl = final_result_pl.filter(pl.col('season') == max_season, pl.col('plays') > 30).drop('season', 'plays')
    final_result_pl = final_result_pl.sort('tdsr', descending=True)

    # For 'cpoe'
    cpoe_mean = final_result_pl.select(pl.col('cpoe').mean()).to_series()[0]
    cpoe_std = final_result_pl.select(pl.col('cpoe').std()).to_series()[0]

    # For 'tdsr'
    tdsr_mean = final_result_pl.select(pl.col('tdsr').mean()).to_series()[0]
    tdsr_std = final_result_pl.select(pl.col('tdsr').std()).to_series()[0]

    # For 'epa'
    epa_mean = final_result_pl.select(pl.col('epa').mean()).to_series()[0]
    epa_std = final_result_pl.select(pl.col('epa').std()).to_series()[0]

    # Step 2: Add z-score columns to the DataFrame
    if scale_metrics:
        final_result_pl = final_result_pl.with_columns([
            ((pl.col('cpoe') - pl.lit(cpoe_mean)) / pl.lit(cpoe_std)).alias('cpoe'),
            ((pl.col('tdsr') - pl.lit(tdsr_mean)) / pl.lit(tdsr_std)).alias('tdsr'),
            ((pl.col('epa') - pl.lit(epa_mean)) / pl.lit(epa_std)).alias('epa')
        ])
    

    return final_result_pl

seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
rosters = pl.from_pandas(nfl.__import_rosters(release="seasonal", years=seasons))
pbp_raw = load_local_pbp(seasons=seasons)
pbp = generate_primary_passers(pbp_raw)
tdsr_df = calculate_tdsr(pbp_df=pbp, min_plays=15, rosters_df=rosters, scale_metrics=False)

print(tdsr_df)

# pbp.select(pl.col('fixed_drive_result')).unique()