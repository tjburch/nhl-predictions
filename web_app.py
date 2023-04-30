import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from pathlib import Path
# Data formatting 
# -------------------------------------------------------------------------------------
team_colors = {
    "Anaheim Ducks": "#F47A38",
    "Arizona Coyotes": "#8C2633",
    "Boston Bruins": "#FFB81C",
    "Buffalo Sabres": "#002654",
    "Calgary Flames": "#C8102E",
    "Carolina Hurricanes": "#CC0000",
    "Chicago Blackhawks": "#CF0A2C",
    "Colorado Avalanche": "#6F263D",
    "Columbus Blue Jackets": "#002654",
    "Dallas Stars": "#00843D",
    "Detroit Red Wings": "#CE1126",
    "Edmonton Oilers": "#FF4C00",
    "Florida Panthers": "#C8102E",
    "Los Angeles Kings": "#111111",
    "Minnesota Wild": "#A6192E",
    "Montreal Canadiens": "#AF1E2D",
    "Nashville Predators": "#FFB81C",
    "New Jersey Devils": "#CE1126",
    "New York Islanders": "#00539B",
    "New York Rangers": "#0038A8",
    "Ottawa Senators": "#C52032",
    "Philadelphia Flyers": "#F74902",
    "Pittsburgh Penguins": "#CFC493",
    "San Jose Sharks": "#006D75",
    "Seattle Kraken": "#001628",
    "St. Louis Blues": "#002F87",
    "Tampa Bay Lightning": "#002868",
    "Toronto Maple Leafs": "#00205B",
    "Vancouver Canucks": "#00205B",
    "Vegas Golden Knights": "#B4975A",
    "Washington Capitals": "#041E42",
    "Winnipeg Jets": "#041E42"
}
current_date = datetime.now().strftime("%Y-%m-%d")
directory_path = Path(__file__).parent
output_df = pd.read_parquet(directory_path / f"data/prepped/{current_date}.pq")

# Web App 
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="NHL Playoff Predictions", layout="wide")
st.title("NHL Playoff Predictions")

st.write("""
Below are predictions for the remainder of the 2023 NHL playoffs. The underlying model assumes that the rate at which each team scores and gives up points are realizations of latent "goal scoring" and "goal suppressing" parameters. Additionally, a factor for home-ice advantage is incorporated. The model is fit to the 2022-2023 regular season data.
""")

# Function to apply the color gradient to table cells
def highlight_cells(val):
    cmap = plt.get_cmap("Reds")  # Reds colormap
    rgba = cmap(val / 100)
    color = mpl.colors.rgb2hex(rgba)

    # Calculate brightness to determine text color
    brightness = (rgba[0] * 299 + rgba[1] * 587 + rgba[2] * 114) / 1000
    text_color = "white" if brightness < 0.5 else "black"
    
    return f"background-color: {color}; color: {text_color};"


# Display the table with formatting and sorting enabled
st.dataframe(
    output_df.style.format("{:.0f}%").applymap(highlight_cells).set_table_styles(
        [{"selector": "th", "props": [("cursor", "pointer")]}
    ]),
)


st.write("""
Model and visuals all produced by Tyler Burch. For more information on the methodology, see [this blog post](tylerjamesburch.com/blog/misc/nhl-predictions)
""")
