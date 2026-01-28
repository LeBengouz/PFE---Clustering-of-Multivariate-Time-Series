import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Clustering - taxi San Francisco - 2

    Ce notebook est la suite d'un premier notebook "Cab_Clustering_distance_based" qui visait à explorer le dataset des taxi de San Francisco et utiliser des méthodes classiques de clustering sur les séries temporelles multivariées qu'il contient. Cependant, ce premier notebook a surtout permis de construire des outils de visualisation, des objectifs et attendus quant au clustering et aussi de soulever des problématiques.

    C'est pourquoi ce notebook vise à explorer les méthodes de clustering "subsequence-based". Nous allons dans un premier temps reprendre certaines fonctions permettant de charger et visualiser les données. Le détail du fonctionnement de celles-ci figure dans le premier notebook.

    ## Pre-processing

    ### Extraction des données
    """)
    return


@app.cell
def _():
    import marimo as mo
    import tslearn
    from tqdm import tqdm
    import pandas as pd


    import numpy as np
    np.random.seed(307)
    SEED = 307
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1) Extraire le dataset par taxi :
    - Liste des taxi
    - Puis dataframe des séries par taxi
    """)
    return


@app.cell
def _(pd):
    import re
    from pathlib import Path


    DATA_DIR = Path("./cabspottingdata")
    _CABS_FILE = DATA_DIR / "_cabs.txt"

    cab_pattern = re.compile(r'<cab id="([^"]+)" updates="(\d+)"/>')

    _cabs = []
    with _CABS_FILE.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = cab_pattern.search(line)
            if m:
                cab_id, updates = m.group(1), int(m.group(2))
                _cabs.append({"cab_id": cab_id, "updates": updates})

    # cabs_df liste les taxi et leur nombre d'updates
    cabs_df = pd.DataFrame(_cabs)
    cabs_df.head(), len(cabs_df)
    return DATA_DIR, cabs_df


@app.cell
def _(DATA_DIR, cabs_df, pd):
    def read_cab_file(cab_id):
        fp = DATA_DIR / f"new_{cab_id}.txt"
        if not fp.exists():
            return pd.DataFrame(columns=["cab_id", "lat", "lon", "occupied", "timestamp"])

        df = pd.read_csv(
            fp,
            sep=r"\s+",
            header=None,
            names=["lat", "lon", "occupied", "timestamp"],
            dtype={"lat": "float64", "lon": "float64", "occupied": "int8", "timestamp": "int64"},
        )
        df.insert(0, "cab_id", cab_id)
        return df

    def load_trips(cabs_df: pd.DataFrame):
        dfs = [read_cab_file(cab_id) for cab_id in cabs_df["cab_id"]]
        df = pd.concat(dfs, ignore_index=True)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df.sort_values(["cab_id", "timestamp"]).reset_index(drop=True)

    _trips_df = load_trips(cabs_df)

    cab_seq_df = (_trips_df.groupby("cab_id", sort=False).agg(lat=("lat", list),lon=("lon", list),occupied=("occupied", list),timestamp=("timestamp",list),).reset_index())

    cab_seq_df.head(), cab_seq_df.shape
    return (cab_seq_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2) En déduire le dataset par trajet :
    """)
    return


@app.cell
def _(pd):
    def build_trips_seq_df(cab_df, min_len= 2):
        voyages = []
        compteur = 0


        for row in cab_df.itertuples(index=False):
            cab_id = row.cab_id
            lat = row.lat
            lon = row.lon
            occupied = row.occupied
            ts = row.timestamp

            # Sécurité: on suppose len alignées, sinon on tronque au minimum
            L = min(len(lat), len(lon), len(occupied), len(ts))
            if L < min_len: # ne pas concerver les tout petits trajets
                continue

            lat = lat[:L]
            lon = lon[:L]
            occupied = occupied[:L]
            ts = ts[:L]

            # Découpe par runs de occupied constant
            start = 0
            current_occupation = occupied[0]

            for i in range(1, L + 1):
                # Si changement d'occupation ou fin de la limite
                if i == L or occupied[i] != current_occupation:
                    end = i
                    segment_length = end - start

                    if segment_length >= min_len:
                        voyages.append({
                            "trip_id": compteur,
                            "cab_id": cab_id,
                            "occupied": current_occupation,
                            "lat": lat[start:end],
                            "lon": lon[start:end],
                            "timestamp": ts[start:end],
                            "t_start": ts[start],
                            "t_end": ts[end - 1],
                        })
                        compteur += 1
                    # Maj nouvelle liste
                    if i < L:
                        start = i
                        current_occupation = occupied[i]

        voyages_df = pd.DataFrame(voyages)

        return voyages_df
    return (build_trips_seq_df,)


@app.cell
def _(build_trips_seq_df, cab_seq_df):
    voyages_df = build_trips_seq_df(cab_seq_df, 2)
    voyages_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation des données

    Ajoutons les outils de visualisation des données en cas de besoin.
    """)
    return


@app.cell
def _():
    # To complete
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clustering subsequence-based

    Dans cette partie, détailler la distinction Shapelets vs Sliding window. Nous voulons nous concentrer sur certaines des approches subsquenced-based dont notamment :
    - MCSCPS (multi-stage clustering and probabilistic forecasting) - 2017
    - SHAPNET (Shapelet-Neural Network Approach for Multivariate Time Series Classification) - 2021
    - CSL (Shapelet-based Framework for Unsupervised Multivariate Time Series Representation Learning) - 2023

    La première approche...


    La seconde avec SHAPNET utilise le concept de shapelet. Cet algorithme de deep learning vise à sélectionner des shapelets via des candidats existants...

    La troisième (CSL) vise à entraîner des représentation des séries shapelet-based à différentes échelles...
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
