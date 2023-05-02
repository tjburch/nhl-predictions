import pymc as pm
import aesara.tensor as at
import pytensor.tensor as pt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import poisson
import multiprocessing as mp
from functools import partial
from pathlib import Path
from datetime import datetime
import importlib

# Import current W/L record
today_str = datetime.now().strftime('%Y-%m-%d')
module_name = f"data.true_wl.{today_str}"
module = importlib.import_module(module_name)
round1_true_wl, round2_true_wl, conf_true_wl, stan_true_wl = (
    module.round1_true_wl or None,
    module.round2_true_wl or None,
    module.conf_true_wl or None,
    module.stan_true_wl or None
)


# Data Injesting
# ------------------------------------------------
def data_injest():
    full_schedule = pd.read_csv(r"/Users/tburch/Documents/github/nhl-hierarchical/end_of_year_schedule.csv")
    team_names = sorted(full_schedule['Visitor'].append(full_schedule['Home']).unique())
    team_mapping = {team_name: idx for idx, team_name in enumerate(team_names)}
    # Preprocess
    data = (full_schedule.rename(columns={"Visitor": "away_team", "G": "away_goals", "Home": "home_team", "G.1": "home_goals", "Unnamed: 5": "game_outcome"})
    .drop(columns=["Date", "Att.", "LOG", "Notes"])
    .assign(
            home_goals_reg=lambda x: x.apply(lambda row: row['home_goals'] if row['game_outcome'] != 'OT' else row['home_goals'] - 1, axis=1),
            away_goals_reg=lambda x: x.apply(lambda row: row['away_goals'] if row['game_outcome'] != 'OT' else row['away_goals'] - 1, axis=1),
            home_goals_ot=lambda x: x.apply(lambda row: 1 if row['game_outcome'] == 'OT' and row['home_goals'] > row['away_goals'] else 0, axis=1),
            away_goals_ot=lambda x: x.apply(lambda row: 1 if row['game_outcome'] == 'OT' and row['home_goals'] < row['away_goals'] else 0, axis=1),
            shootout_winner=lambda x: x.apply(lambda row: 1 if row['game_outcome'] == 'SO' and row['home_goals'] > row['away_goals'] else (0 if row['game_outcome'] == 'SO' and row['home_goals'] < row['away_goals'] else None), axis=1))
    )
    return data

# Model Definition
# ------------------------------------------------
def overtime_goals_likelihood(observed_ot_h_goals, observed_ot_a_goals, ot_h_theta, ot_a_theta):
    allowed_outcomes = [(0, 0), (1, 0), (0, 1)]
    likelihoods = []

    for h_goals, a_goals in allowed_outcomes:
        h_likelihood = pm.logp(pm.Poisson.dist(mu=ot_h_theta), observed_ot_h_goals * h_goals)
        a_likelihood = pm.logp(pm.Poisson.dist(mu=ot_a_theta), observed_ot_a_goals * a_goals)
        likelihoods.append(h_likelihood + a_likelihood)

    return pm.math.logsumexp(pm.math.stack(likelihoods), axis=0)


def model_fit(data):
        
    home_idx, teams = pd.factorize(data["home_team"], sort=True)
    away_idx, _ = pd.factorize(data["away_team"], sort=True)

    coords = {
        "team": teams,
        "match": np.arange(len(data)),
    }

    with pm.Model(coords=coords) as model:
        # Global model parameters
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        home = pm.Normal("home", mu=0, sigma=0.2)

        # Hyperpriors for attacks and defs
        sd_att = pm.HalfCauchy("sd_att", 0.2)
        sd_def = pm.HalfCauchy("sd_def", 0.2)

        # Team-specific model parameters
        atts_star = pm.Normal("atts_star", mu=0, sigma=sd_att, dims="team")
        defs_star = pm.Normal("defs_star", mu=0, sigma=sd_def, dims="team")

        # Demeaned team-specific parameters
        atts = pm.Deterministic("atts", atts_star - at.mean(atts_star), dims="team")
        defs = pm.Deterministic("defs", defs_star - at.mean(defs_star), dims="team")

        # Expected goals for home and away teams during regulation
        home_theta = at.exp(intercept + home + atts[home_idx] - defs[away_idx])
        away_theta = at.exp(intercept + atts[away_idx] - defs[home_idx])

        # Likelihood (Poisson distribution) for regulation goals
        home_points = pm.Poisson("home_points", mu=home_theta, observed=data['home_goals'], dims="match")
        away_points = pm.Poisson("away_points", mu=away_theta, observed=data['away_goals'], dims="match")

        # Overtime and shootout deterministics
        overtime = data['home_goals'] == data['away_goals']
        shootout = (data['home_goals_ot'] == data['away_goals_ot']) & overtime

        # Expected goals for home and away teams during overtime (scaled down by 1/12)
        ot_home_theta = home_theta * (1 / 12)
        ot_away_theta = away_theta * (1 / 12)

        # Likelihood (custom likelihood function) for overtime goals
        if overtime.sum() > 0:
            pm.Potential("ot_goals_constraint",
                        overtime_goals_likelihood(data.home_goals_ot, data.away_goals_ot, ot_home_theta, ot_away_theta))

        # Shootout model (conditioned on games that went to shootout)
        so_coeff_o = pm.Normal("so_coeff_o", mu=0, sigma=1, dims="team")  # Offensive shootout coefficient
        so_coeff_d = pm.Normal("so_coeff_d", mu=0, sigma=1, dims="team")  # Defensive shootout coefficient
        so_coeff_h = pm.Normal("so_coeff_h", mu=0, sigma=1)  # Home advantage coefficient
        so_intercept = pm.Normal("so_intercept", mu=0, sigma=1)  # Intercept term

        so_logit = (so_intercept +
                    so_coeff_o[home_idx[shootout]] - so_coeff_o[away_idx[shootout]] +
                    so_coeff_d[home_idx[shootout]] - so_coeff_d[away_idx[shootout]] +
                    so_coeff_h * home)

        if shootout.sum() > 0:
            so_prob = pm.math.invlogit(so_logit)
            shootout_winner = pm.Bernoulli("shootout_winner", p=so_prob, observed=data['shootout_winner'][shootout])

        trace = pm.sample(4000, tune=3000)
    return model, trace


def post_flt(array, team_name=None):
    if team_name is None:
        return array.values.reshape(-1)
    else:
        return array.sel(team=team_name).values.reshape(-1)
    

def get_matchup_prob(trace, home_team, away_team, n_samples=100):
    
    # Get the posterior samples for the specified teams
    home_atts = trace.posterior['atts'].sel(team=home_team).values.flatten()
    away_atts = trace.posterior['atts'].sel(team=away_team).values.flatten()
    home_defs = trace.posterior['defs'].sel(team=home_team).values.flatten()
    away_defs = trace.posterior['defs'].sel(team=away_team).values.flatten()

    # Sample from the posterior distribution
    intercept_samples = np.random.choice(trace.posterior['intercept'].values.flatten(), n_samples)
    home_samples = np.random.choice(trace.posterior['home'].values.flatten(), n_samples)
    home_atts_samples = np.random.choice(home_atts, n_samples)
    away_atts_samples = np.random.choice(away_atts, n_samples)
    home_defs_samples = np.random.choice(home_defs, n_samples)
    away_defs_samples = np.random.choice(away_defs, n_samples)

    home_theta_samples = np.exp(intercept_samples + home_samples + home_atts_samples - away_defs_samples)
    away_theta_samples = np.exp(intercept_samples + away_atts_samples - home_defs_samples)

    # Scale down the theta values for 20-minute overtime periods
    ot_home_theta_samples = home_theta_samples * (1 / 3)
    ot_away_theta_samples = away_theta_samples * (1 / 3)

    # Compute the probability of home team winning using the sampled parameters
    win_prob_samples = []
    for home_theta, away_theta, ot_home_theta, ot_away_theta in zip(home_theta_samples, away_theta_samples, ot_home_theta_samples, ot_away_theta_samples):
        # Assume no more than 8 goals
        home_win_prob_regulation = sum([poisson.pmf(home_goals, home_theta) * sum([poisson.pmf(away_goals, away_theta) for away_goals in range(home_goals)]) for home_goals in range(9)])
        home_win_prob_overtime = sum([poisson.pmf(home_goals, ot_home_theta) * sum([poisson.pmf(away_goals, ot_away_theta) for away_goals in range(home_goals)]) for home_goals in range(1, 9)])
        total_home_win_prob = home_win_prob_regulation + (1 - home_win_prob_regulation) * home_win_prob_overtime
        win_prob_samples.append(total_home_win_prob)
    
    mean_win_prob = np.mean(win_prob_samples)

    return mean_win_prob

def simulate_playoff_round(matchups, trace, team_points, starting_wins=None):
    results = []

    if not isinstance(matchups[0], tuple) and not isinstance(matchups[0], list):
        matchups = [matchups]

    for team_a, team_b in matchups:
        if starting_wins is not None:
            team_a_wins, team_b_wins = starting_wins[(team_a, team_b)]['team_a_wins'], starting_wins[(team_a, team_b)]['team_b_wins']
        else:
            team_a_wins = 0
            team_b_wins = 0

        if team_a_wins < 4 and team_b_wins < 4:
            team_a_home_win_prob = get_matchup_prob(trace, team_a, team_b)
            team_b_home_win_prob = get_matchup_prob(trace, team_b, team_a)
        else:
            team_a_home_win_prob = np.nan
            team_b_home_win_prob = np.nan
 
        # Simulate games with the 2-2-1-1-1 format
        for game in range(1, 2 * 4):
            if team_a_wins >= 4 or team_b_wins >= 4:
                break

            is_home_team_a = (
                (game <= 2) or (game == 5) or (game == 7)
            )

            if is_home_team_a:
                if np.random.random() < team_a_home_win_prob:
                    team_a_wins += 1
                else:
                    team_b_wins += 1
            else:
                if np.random.random() < team_b_home_win_prob:
                    team_b_wins += 1
                else:
                    team_a_wins += 1



        winner = team_a if team_a_wins > team_b_wins else team_b
        games_played = team_a_wins + team_b_wins
        results.append({
            'team_a': team_a,
            'team_b': team_b,
            'team_a_wins': team_a_wins,
            'team_b_wins': team_b_wins,
            'winner': winner,
            'games_played': games_played,
            'team_a_home_prob': team_a_home_win_prob,
            'team_b_home_prob': team_b_home_win_prob,
            })

    if len(matchups) == 1:
        return results[0]

    return results


team_points = {
    'Boston Bruins': 135,
    'Carolina Hurricanes': 113,
    'New Jersey Devils': 112,
    'Vegas Golden Knights': 111,
    'Toronto Maple Leafs': 111,
    'Colorado Avalanche': 109,
    'Edmonton Oilers': 109,
    'Dallas Stars': 108,
    'New York Rangers': 107,
    'Los Angeles Kings': 104,
    'Minnesota Wild': 103,
    'Seattle Kraken': 100,
    'Tampa Bay Lightning': 98,
    'Winnipeg Jets': 95,
    'Calgary Flames': 93,
    'New York Islanders': 93,
    'Florida Panthers': 92,
}

def run_playoff_sim(
        trace,
        round_1_wl=None,
        round_2_wl=None,
        conf_wl=None,
        stan_wl=None
):
    eastern_conference_matchups = [
        ('Boston Bruins', 'Florida Panthers'),
        ('Toronto Maple Leafs', 'Tampa Bay Lightning'),
        ('Carolina Hurricanes', 'New York Islanders'),
        ('New Jersey Devils', 'New York Rangers')
    ]
    eastern_conference_matchups = [sorted(s) for s in eastern_conference_matchups]
    western_conference_matchups = [
        ('Vegas Golden Knights', 'Winnipeg Jets'),
        ('Edmonton Oilers', 'Los Angeles Kings'),
        ('Colorado Avalanche', 'Seattle Kraken'),
        ('Dallas Stars', 'Minnesota Wild')
    ]
    western_conference_matchups = [sorted(s) for s in western_conference_matchups]


    # ROUND 1
    # --------------
    east_round_1_results = simulate_playoff_round(eastern_conference_matchups, trace, team_points, round_1_wl)
    west_round_1_results = simulate_playoff_round(western_conference_matchups, trace, team_points, round_1_wl)

    east_round_1_winners = [result['winner'] for result in east_round_1_results]
    west_round_1_winners = [result['winner'] for result in west_round_1_results]

    round_1_df = pd.DataFrame(east_round_1_results + west_round_1_results)

    # ROUND 2
    # --------------
    east_round_2_matchups = [tuple(east_round_1_winners[0:2]), tuple(east_round_1_winners[2:])]
    west_round_2_matchups = [tuple(west_round_1_winners[0:2]), tuple(west_round_1_winners[2:])]

    east_round_2_results = simulate_playoff_round(east_round_2_matchups, trace, team_points, round_2_wl)
    west_round_2_results = simulate_playoff_round(west_round_2_matchups, trace, team_points, round_2_wl)

    round_2_df = pd.DataFrame(east_round_2_results + west_round_2_results)

    east_round_2_winners = [result['winner'] for result in east_round_2_results]
    west_round_2_winners = [result['winner'] for result in west_round_2_results]

    # CONFERENCE FINALS
    # --------------
    east_conference_finals_matchup = tuple(east_round_2_winners)
    west_conference_finals_matchup = tuple(west_round_2_winners)

    east_conference_finals_result = simulate_playoff_round(east_conference_finals_matchup, trace, team_points, conf_wl)
    west_conference_finals_result = simulate_playoff_round(west_conference_finals_matchup, trace, team_points, conf_wl)

    conf_df = pd.DataFrame([east_conference_finals_result, west_conference_finals_result])

    east_conference_winner = east_conference_finals_result['winner']
    west_conference_winner = west_conference_finals_result['winner']

    # STANLEY CUP
    # --------------
    stanley_cup_finals_matchup = (east_conference_winner, west_conference_winner)

    # Simulate the Stanley Cup finals
    stanley_cup_finals_result = simulate_playoff_round(stanley_cup_finals_matchup, trace, team_points, stan_wl)
    stanley_cup_winner = stanley_cup_finals_result['winner']
    stanley_df = pd.DataFrame([stanley_cup_finals_result])
    return (round_1_df, round_2_df, conf_df, stanley_df)

def run_simulation(_, trace, round_1_wl, round_2_wl, conf_wl, stan_wl):
    return run_playoff_sim(trace, round_1_wl=round_1_wl, round_2_wl=round_2_wl, conf_wl=conf_wl, stan_wl=stan_wl)

def generate_all_sims(trace, round1_true_wl, round2_true_wl, conf_true_wl, stan_true_wl, num_simulations=5):
    print("Starting Simulations")
    with mp.Pool(mp.cpu_count()) as pool:
        run_sim_with_wl = partial(
            run_simulation, 
            trace=trace, 
            round_1_wl=round1_true_wl,
            round_2_wl=round2_true_wl,
            conf_wl=conf_true_wl,
            stan_wl=stan_true_wl
            )
        results = list(tqdm(pool.imap(run_sim_with_wl, range(num_simulations)), total=num_simulations))

    all_round_1_dfs = pd.concat([res[0] for res in results])
    all_round_2_dfs = pd.concat([res[1] for res in results])
    all_conf_dfs = pd.concat([res[2] for res in results])
    all_stanley_dfs = pd.concat([res[3] for res in results])

    current_date = datetime.now().strftime("%Y-%m-%d")
    directory_path = Path(__file__).parent
    output_dir = directory_path / Path(f"./data/sim_output/{current_date}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_round_1_dfs.to_parquet(output_dir / "round_1.pq")
    all_round_2_dfs.to_parquet(output_dir / "round_2.pq")
    all_conf_dfs.to_parquet(output_dir / "conference_finals.pq")
    all_stanley_dfs.to_parquet(output_dir / "stanley_cup_finals.pq")


if __name__ == "__main__":

    data = data_injest()
    model, trace = model_fit(data)
    print("Model fit")
    generate_all_sims(trace, round1_true_wl, round2_true_wl, conf_true_wl, stan_true_wl, num_simulations=500)
