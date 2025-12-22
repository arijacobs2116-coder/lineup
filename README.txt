lineupstats

A Streamlit app for analyzing basketball 5-player lineup data.

The app groups unique lineups (player order does not matter), applies a minimum possessions filter, and ranks lineups using offensive and defensive statistics.

FEATURES
- Upload lineup CSV files
- Automatic 5-player lineup grouping
- Separate offense and defense tabs
- Lineup ranking by statistical categories
- Minimum possessions threshold slider
- Ignores offensive/defensive possession columns in rankings

REQUIREMENTS
- Python 3.9+
- streamlit
- pandas
- numpy

INSTALL
pip install -r requirements.txt

RUN
streamlit run app.py
