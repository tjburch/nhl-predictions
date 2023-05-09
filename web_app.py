import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, date
from pathlib import Path
import glob
import re
import pathlib
from bs4 import BeautifulSoup
import logging
import shutil

def inject_ga():
    GA_ID = "google_analytics"

    GA_JS = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-WQ3JL3NLY9"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'G-WQ3JL3NLY9');
    </script>
    """

    # Insert the script in the head tag of the static template inside your virtual
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=GA_ID):  # if cannot find tag
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)  # recover from backup
        else:
            shutil.copy(index_path, bck_index)  # keep a backup
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + GA_JS)
        index_path.write_text(new_html)


def get_date_str_and_directory_path():
    directory_path = Path(__file__).parent

    def is_valid_date_format(date_string):
        pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
        return bool(pattern.match(date_string))

    parquet_files = glob.glob(str(directory_path / "data/prepped/*.pq"))
    dates = [
        datetime.strptime(Path(file).stem, "%Y-%m-%d")
        for file in parquet_files
        if is_valid_date_format(Path(file).stem)
    ]
    latest_date = max(dates)
    date_str = latest_date.strftime("%Y-%m-%d")

    return date_str, directory_path, dates


def load_data_for_date(date_str, directory_path):
    output_df = pd.read_parquet(directory_path / f"data/prepped/{date_str}.pq")
    file_path = directory_path / Path(f"data/prepped/matchups_{date_str}.pq")
    if file_path.exists():
        matchup_prob_df = pd.read_parquet(file_path)
        matchup_prob_df['Home Win Probability'] = matchup_prob_df['Home Win Probability'].astype(int).astype(str) + '%'
        matchup_prob_df['Away Win Probability'] = matchup_prob_df['Away Win Probability'].astype(int).astype(str) + '%'
    else:
        matchup_prob_df = None
    return output_df, matchup_prob_df

def highlight_cells(val):
    cmap = plt.get_cmap("Reds")  # Reds colormap
    rgba = cmap(val / 100)
    color = mpl.colors.rgb2hex(rgba)

    # Calculate brightness to determine text color
    brightness = (rgba[0] * 299 + rgba[1] * 587 + rgba[2] * 114) / 1000
    text_color = "white" if brightness < 0.5 else "black"
    
    return f"background-color: {color}; color: {text_color};"


# Data formatting 
# -------------------------------------------------------------------------------------
def predictions_page(directory_path, dates):
    # Add a dropdown box for selecting the date
    selected_date = st.selectbox("Load projections from date:", options=[d.strftime("%Y-%m-%d") for d in sorted(dates, reverse=True)], index=0)

    # Load the data for the selected date
    output_df, matchup_prob_df = load_data_for_date(selected_date, directory_path)

    # Display the table with formatting and sorting enabled
    st.dataframe(
        output_df.style.format("{:.0f}%").applymap(highlight_cells).set_table_styles(
            [{"selector": "th", "props": [("cursor", "pointer")]}
        ]),
    )

    # Add daily matchup probabilities
    if matchup_prob_df is not None:
        # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """
        st.title("Daily Matchup Probabilities")
        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(matchup_prob_df)


def about_model_page(directory_path):
    st.subheader("About the Model")
    st.write("""
    This page contains predictions for the 2022-2023 NHL playoffs. At the core of the model, we assume that goal scoring is an observation of a Poisson distributed random variable. Poisson distributions are typically used to predict the number of events that occur in a fixed time and are controlled by a single parameter, which governs the rate. The rate used has 3 components: a latent *goal-scoring* and *goal-suppressing* parameter, unique for each team, and a global home-ice advantage. In each game, the home score rate parameter is the sum of the home team's goal-scoring ability, the away team's goal-suppressing ability, and the home-ice advantage. The away score rate parameter is the sum of the away team's goal-scoring ability and the home team's goal-suppressing ability.\n
    This model is fit on data from the 2022-2023 regular season to infer each team's *goal-scoring* and *goal-suppressing* parameters and the home-ice advantage. Once that is done, 500 simulations are run based on the current win-loss records each day to model the remainder of the playoffs.

    Given this configuration, the following parameters are found:
    """)

    local_image_path = directory_path / Path(r"visualizations/parameter_plane.png")
    st.image(str(local_image_path), caption="Inferred parameters for each team", width=600)

    st.write("""
    More specifically, a Bayesian hierarchical model is used for the goal-scoring and goal-suppressing parameters, which has two added benefits. The first is Bayesian *shrinkage*, which improves inference accuracy by considering overall trends in the data and sharing information between teams rather than considering each independently. The second is built-in uncertainty quantification. You can see the model's uncertainty in the goal-scoring and goal-suppressing parameters below.
    """)

    st.image(str(directory_path / Path(r"visualizations/goal_creation.png")), caption="Goal-Creation Parameter",width=800)
    st.image(str(directory_path / Path(r"visualizations/goal_suppression.png")), caption="Goal-Suppression Parameter",width=800)

    st.write("""
    Lastly, the home-ice advantage parameter is found to provide just over one goal's worth of benefit to the home team (95% CI is 1.011 to 1.101 goals).
    """)
    st.image(str(directory_path / Path(r"visualizations/home_ice.png")), caption="Home Ice Parameter", width=600)

    st.write("For more technical information about the model, see [this blog post](https://tylerjamesburch.com/blog/misc/nhl-predictions)")

def main():

    latest_date_str, directory_path, dates = get_date_str_and_directory_path()

    st.set_page_config(
        page_title="NHL Playoff Predictions",
        layout="wide",
    )
    inject_ga()
    st.title("NHL Playoff Predictions")
    st.write(f"Last updated: {latest_date_str}")

    tabs = st.tabs(["Predictions", "About the Model"])
    predictions_tab, about_model_tab = tabs

    with predictions_tab:
        predictions_page(directory_path, dates)

    with about_model_tab:
        about_model_page(directory_path)


    st.markdown("-----")
    st.write("""
    Model and visuals all produced by [Tyler Burch](https://tylerjamesburch.com/)
    """)
    st.caption("""
    Thanks for visiting - if you enjoy this content, please [help me keep the app alive!](https://bmc.link/tylerjamesburch)
            """)

if __name__ == "__main__":
    main()