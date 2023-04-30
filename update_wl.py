import pandas as pd
from datetime import datetime, timedelta
import os
import hockey_scraper
from collections import defaultdict

ROUND_1_START_DATE = "2023-04-16"
ROUND_2_START_DATE = "2023-05-01"
CONF_FINAL_START_DATE = "2023-05-15"
STANLEY_CUP_START_DATE = "2023-05-29"


def scrape_schedule_to_dict(start_date, end_date):
    team_names = {
        'EDM': 'Edmonton Oilers',
        'L.A': 'Los Angeles Kings',
        'N.J': 'New Jersey Devils',
        'NYR': 'New York Rangers',
        'TOR': 'Toronto Maple Leafs',
        'T.B': 'Tampa Bay Lightning',
        'BOS': 'Boston Bruins',
        'FLA': 'Florida Panthers',
        'DAL': 'Dallas Stars',
        'MIN': 'Minnesota Wild',
        'COL': 'Colorado Avalanche',
        'SEA': 'Seattle Kraken',
        'CAR': 'Carolina Hurricanes',
        'NYI': 'New York Islanders',
        'VGK': 'Vegas Golden Knights',
        'WPG': 'Winnipeg Jets',
    }
    df = hockey_scraper.scrape_schedule(start_date, end_date)
    completed_games = df[df['status'] == 'Final']

    # Initialize an empty defaultdict
    matchup_wins = defaultdict(lambda: {'team_a': '', 'team_b': '', 'team_a_wins': 0, 'team_b_wins': 0})

    for index, row in completed_games.iterrows():
        home_team = team_names[row['home_team']]
        away_team = team_names[row['away_team']]

        # Create a tuple with team names in alphabetical order
        matchup_key = tuple(sorted((home_team, away_team)))

        # Update the team names in the dictionary
        matchup_wins[matchup_key]['team_a'] = matchup_key[0]
        matchup_wins[matchup_key]['team_b'] = matchup_key[1]

        # Increment the win count for the winning team
        if row['home_score'] > row['away_score']:
            winning_team = home_team
        else:
            winning_team = away_team

        if winning_team == matchup_key[0]:
            matchup_wins[matchup_key]['team_a_wins'] += 1
        else:
            matchup_wins[matchup_key]['team_b_wins'] += 1

    # Convert the defaultdict to a regular dictionary
    return dict(matchup_wins)

# Define a helper function to save the dictionary to a file
def save_dict_to_file(round_results_dict, date_string, round_name):
    file_path = f"data/true_wl/{date_string}.py"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"{round_name} = {round_results_dict}\n")

today = datetime.now().strftime("%Y-%m-%d")
round1_dict = scrape_schedule_to_dict(ROUND_1_START_DATE, ROUND_2_START_DATE)
round2_dict = scrape_schedule_to_dict(ROUND_2_START_DATE, CONF_FINAL_START_DATE)
conf_dict = scrape_schedule_to_dict(CONF_FINAL_START_DATE, STANLEY_CUP_START_DATE)
stan_dict = scrape_schedule_to_dict(STANLEY_CUP_START_DATE, "2023-07-01")

save_dict_to_file(round1_dict, today, "round1_true_wl")
save_dict_to_file(round2_dict, today, "round2_true_wl")
save_dict_to_file(conf_dict, today, "conf_true_wl")
save_dict_to_file(stan_dict, today, "stan_true_wl")