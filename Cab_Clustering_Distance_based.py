import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering - taxi San Francisco - 1

    L'objectif de ce notebook est la prise en main de marimo et du jeu de données sfcab. Il s'agit ici de ploter les données, les comprendre. Puis dans un second temps, l'objectif sera d'aborder certaines manière de réaliser du clustering. L'objectif est d'aborder certains algorithmes classiques et des applications naïves à des données multi-variés puis les comparer avec des algorithmes potentiellement plus adaptés selon les survey dans le domaine. Cette comparaison se fera dans d'autres notebook suivant ce premier jet.

    Les algorithmes abordés dans ce notebook sont des variantes de k-means adaptés aux séries multivariés. Il s'agit de deux premières pipelines simple de la catégorie des clustering "distance based". L'objectif est de relever les performances de ce premiers algorithmes, comprendre comment ils s'adaptent aux cas multivarié et surtout soulever les problématiques principales liées au clustering de certaines séries temporelles multivariées.


    Le jeu de données utilisé est celui de "CRAWDAD epfl/mobility". Il concerne les taxi de San Francisco.

    ## Traitement des données
    """)
    return


@app.cell
def _():
    import marimo as mo
    import tslearn
    from tqdm import tqdm


    import numpy as np
    np.random.seed(307)
    return mo, np, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importer les Données

    Le jeu de donnée se trouve en local dans le dossier "cabspottingdata" placé où on le souhaite. Il convient alors d'adapter le path des données en fonction. Ce dossier contient des fichiers .txt pour chaque taxi et le fichier "_cabs.txt". Celui-ci contient la liste des taxi, leur id et le nombre d'update qu'ils présentent. On peut tout de suite remarquer que ce nombre n'est pas égal entre tous les taxi.

    Utilisons "_cabs.txt" pour récupérer les cab_id afin de lire leurs données et remplir un dataset. Chaque ligne correspondra à une mesure d'un taxi. Ainsi, on extrait l'id du taxi, le timestamp de la mesure, la valeur de latitude et longitude, et si le taxi est occupé.
    """)
    return


@app.cell
def _():
    import re
    from pathlib import Path
    import pandas as pd

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
    return DATA_DIR, cabs_df, pd


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
    On peut observer que nous avons donc 536 lignes correspondant aux 536 taxi. Chaque feature est sous la forme d'une liste. Il y a un une mesure de chaque feature à chaque pas de temps. Nous avons donc 536 séries temporelles multivariées puisqu'en se concentrant sur latitude, longtitude et occupied, il y a 3 mesures à chaque fois.

    Dans un premier temps vérifions que la longueur des données corresponde.
    """)
    return


@app.cell
def _(cab_seq_df):
    def count_bad_sequence_lengths(seq_df):
        bad_count = 0
        for _, row in seq_df.iterrows():
            L = len(row["lat"])
            if not (
                len(row["lon"]) == L and
                len(row["occupied"]) == L and
                len(row["timestamp"]) == L
            ):
                bad_count += 1
        return bad_count

    _n_bad = count_bad_sequence_lengths(cab_seq_df)
    print("Nombre de séquences décalées :", _n_bad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Il n'y a pas de valeurs manquantes sur ce dataset ! On peut de plus, vérifier avec un exemple, que le nombre d'update annoncé correspond bien à la longueur des séquences étudiées.
    """)
    return


@app.cell
def _(cab_seq_df, cabs_df):
    print("id du taxi :", cab_seq_df["cab_id"][1])
    print("Longueur des séries", len(cab_seq_df["lat"][1]))
    print("Nombre d'updates affichées", cabs_df.loc[cabs_df["cab_id"] == "abcoij", "updates"].iloc[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affichage et analyses des données

    ### Affichage des courbes

    Commençons par afficher simplement les courbes des séries temporelles dont nous diposons. Nous affichons seulement 2 exemples pour les observer.
    """)
    return


@app.cell
def _(cab_seq_df):
    import matplotlib.pyplot as plt

    def get_plot_multivariate_series(seq_df, cab_id: str, max_points: int | None = 2000):
        row = seq_df.loc[seq_df["cab_id"] == cab_id]
        if row.empty:
            raise ValueError(f"Le taxi '{cab_id}' est introuvable dans le dataframe séquentiel.")
        row = row.iloc[0]

        lat = row["lat"]
        lon = row["lon"]
        occ = row["occupied"]

        n = len(lat)
        if max_points is not None:
            n = min(n, max_points)

        x = list(range(n))

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharex=True, constrained_layout=True)
        fig.suptitle(f"Taxi {cab_id} — séries temporelles par index (n={n})", y=1.02)

        axes[0].plot(x, lat[:n], linewidth=1)
        axes[0].set_title("Latitude")
        axes[0].set_xlabel("index")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(x, lon[:n], linewidth=1)
        axes[1].set_title("Longitude")
        axes[1].set_xlabel("index")
        axes[1].grid(True, alpha=0.25)

        axes[2].step(x, occ[:n], where="post", linewidth=1)
        axes[2].set_title("Occupied")
        axes[2].set_xlabel("index")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.25)

        return fig


    _fig1 = get_plot_multivariate_series(cab_seq_df, cab_seq_df["cab_id"].iloc[0])
    _fig2 = get_plot_multivariate_series(cab_seq_df, cab_seq_df["cab_id"].iloc[1])

    _fig1, _fig2
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut remarquer sur ces deux exemples des comportements différents.
    - Le premier taxi a eu une forte activité ce jour-ci comme en témoignent les courbes : il y a beaucoup de variations sur l'ensemble des 3 courbes.
    - Le second taxi n'a eu qu'une petit période d'activité.
    - La courbe "occupied" n'apporte pas d'information sur le trajet. Cependant les changements entre 0 et 1 peuvent permettre d'indiquer s'il y a eu une activité ou non.

    Cela permet de relever des intérrogations.
    - Est-il pertienent de conserver la donnée occupied ?
    - Sur quelles séries voulons-nous réaliser un clustering

    Cette seconde question est importante. Il semblerait naturel de vouloir réaliser le clustering sur l'entièreté des séries. Mais comme en témoignent ces deux courbes, une abscence de changement peut témoigner d'une inactivité. Est-ce pertinent de réaliser le clustering sur 1 journée de trajet ? Autrement, nous pourrions découper les séries par trajet : en fonction des changements de valeur de "occupied". Cela permettrait alors de distinguer des groupes de trajets à une plus petite échelle.

    Cette réflexion reste encore à résoudre. Dans les deux cas, nous sommes face à un problème de clustering pour lequel le nombre de point de chaque séries n'est pas le même. Notamment pour les mesures de trajets, il peut y avoir des séries à 3 points comment à 20 selon la durée du trajet. Nous pourrons alors évaluer si les algorithmes de clustering gèrent bien ces différences et en plus les valorisent.

    ### Affichage des trajets

    Essayons d'afficher les trajets effectués par certains taxis sur une carte de San-Francisco afin d'analyser leur déplacement sur une carte.
    """)
    return


@app.cell
def _():
    import plotly.express as px
    return (px,)


@app.cell
def _(np, pd):
    import plotly.graph_objects as go

    def estimate_zoom_from_window(window):
        marge_latitude = max(window["north"] - window["south"], 1e-6)
        marge_longitude = max(window["east"] - window["west"], 1e-6)
        marge = max(marge_latitude, marge_longitude)

        zoom = 10 - np.log2(marge / 0.10)
        return float(np.clip(zoom, 3, 18))

    def make_trip_map_from_seq(df, cab_id, max_points=None, trip_idx=None, trip_id=None):
        # récupérer les séries considérées ici
        rows = df.loc[df["cab_id"] == cab_id]
        if rows.empty:
            raise ValueError(f"Taxi nommé '{cab_id}',introuvable dans le dataframe entré")
        if trip_id is not None and "trip_id" in df.columns:
            rows = rows.loc[rows["trip_id"] == trip_id]
            if rows.empty:
                raise ValueError(f"trip_id={trip_id} introuvable pour cab_id='{cab_id}'")
            row = rows.iloc[0]
        elif trip_idx is not None:
            if trip_idx < 0 or trip_idx >= len(rows):
                raise ValueError(f"trip_idx={trip_idx} hors limites (0..{len(rows)-1}) pour cab_id='{cab_id}'")
            row = rows.iloc[trip_idx]
        else:
            row = rows.iloc[0] # pour avoir juste les valeurs

        occ_value = None
        if trip_id is not None and "occupied" in row:
            # puisque les valeurs sont par définition constante, on en prend une
            occ_value = int(pd.Series(row["occupied"]).iloc[0])

        suffix = ""

        if occ_value is not None:
            if occ_value == 0:
                occ_text = "No client"
            else:
                occ_text = "With client"
            suffix += f" — {occ_text}"

        if trip_id is not None:
            suffix += f" — trip_id={trip_id}"


        # Series nécessaires pour Plotly
        latitude = pd.Series(row["lat"])
        longitude = pd.Series(row["lon"])
        occupied = pd.Series(row["occupied"])

        # si valeurs extremes
        mask = latitude.notna() & longitude.notna() & latitude.between(-90, 90) & longitude.between(-180, 180)
        latitude = latitude[mask].reset_index(drop=True)
        longitude = longitude[mask].reset_index(drop=True)
        occupied = occupied[mask].reset_index(drop=True)



        # Limiter affichage pour la lisibilité
        if max_points is not None and len(latitude) > max_points:
            latitude = latitude.iloc[:max_points]
            longitude = longitude.iloc[:max_points]
            occupied = occupied[:max_points]

        if len(latitude) < 2:
            raise ValueError("Pas assez de points valides pour tracer un trajet.")

        # Calcul fenêtre avec marge pour joli affichage
        max_lat = latitude.max()
        min_lat = latitude.min()
        max_lon = longitude.max()
        min_lon = longitude.min()

        marge_latitude = (max_lat - min_lat) * 0.1
        marge_longitude = (max_lon - min_lon) * 0.1


        window = dict(
            west=min_lon - marge_longitude,
            east=max_lon + marge_longitude,
            south=min_lat - marge_latitude,
            north=max_lat + marge_latitude,
        )
        center = {"lat": (window["south"] + window["north"]) / 2, "lon": (window["west"] + window["east"]) / 2}
        zoom = estimate_zoom_from_window(window)


        fig = go.Figure()

        fig.add_trace(
            go.Scattermapbox(
                lat=latitude, lon=longitude,
                mode="lines",
                name="trajet",
                line={"width": 2},
            )
        )

        fig.add_trace(
            go.Scattermapbox(
                lat=latitude, lon=longitude,
                mode="markers",
                name="mesures",
                marker={"size": 5},
            )
        )

        fig.add_trace(
            go.Scattermapbox(
                lat=[latitude.iloc[0]], lon=[longitude.iloc[0]],
                mode="markers", name="départ",
                marker={"size": 10},
            )
        )

        fig.add_trace(
            go.Scattermapbox(
                lat=[latitude.iloc[-1]], lon=[longitude.iloc[-1]],
                mode="markers", name="arrivée",
                marker={"size": 10},
            )
        )

        fig.update_layout(
            title=f"Trajet taxi {cab_id}{suffix} — points affichés: {len(latitude)}",
            margin={"l": 0, "r": 0, "t": 35, "b": 0},
            legend={"orientation": "h"},
            mapbox=dict(
                style="open-street-map",
                bounds=window,     
            ),
        )

        return fig
    return estimate_zoom_from_window, make_trip_map_from_seq


@app.cell
def _(mo):
    mo.md(r"""
    Sélectionnons alors des taxi d'exemple.
    """)
    return


@app.cell
def _(cab_seq_df):
    example_cab_id = cab_seq_df["cab_id"].iloc[1]
    return (example_cab_id,)


@app.cell
def _(cab_seq_df, example_cab_id, make_trip_map_from_seq):
    trip_fig = make_trip_map_from_seq(cab_seq_df, example_cab_id, max_points=20000)
    trip_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Voici quelques observations :
    - Il est difficile d'observer les trajets avec l'ensemble des points de la journée !
    - Les taxi suivent les routes ce qui est fortement visible ici puisque les villes états-uniennes forment une grille. Cependant, lorsqu'on zoom la précision des mesure peut placer les voitures dans des batiments. La précision est relative.
    - Lorsque le taxi est statique, les points se cumulent à un endroit.
    """)
    return


@app.cell
def _(estimate_zoom_from_window, lat, pd, px):
    def make_trip_map_from_seq_simple(df, cab_id, max_points=None, trip_idx=None, trip_id=None):
        # récupérer les séries considérées ici
        rows  = df.loc[df["cab_id"] == cab_id]
        if rows .empty:
            raise ValueError(f"Taxi nommé '{cab_id}',introuvable dans le dataframe entré")

        # choisir une ligne :
        # - si trip_id est fourni et existe -> on filtre
        # - sinon si trip_idx est fourni -> on prend le k-ième trajet
        # - sinon -> première ligne car juste 1 dans le dataset
        if trip_id is not None and "trip_id" in df.columns:
            rows = rows.loc[rows["trip_id"] == trip_id]
            if rows.empty:
                raise ValueError(f"trip_id={trip_id} introuvable pour cab_id='{cab_id}'")
            row = rows.iloc[0]
        elif trip_idx is not None:
            if trip_idx < 0 or trip_idx >= len(rows):
                raise ValueError(f"trip_idx={trip_idx} hors limites (0..{len(rows)-1}) pour cab_id='{cab_id}'")
            row = rows.iloc[trip_idx]
        # Cas sans trip_idx
        else:
            row = rows.iloc[0] # pour avoir juste les valeurs

        occ_value = None
        if trip_id is not None and "occupied" in row:
            # puisque les valeurs sont par définition constante, on en prend une
            occ_value = int(pd.Series(row["occupied"]).iloc[0])


        # Series nécessaires pour Plotly
        latitude = pd.Series(row["lat"])
        longitude = pd.Series(row["lon"])
        occupied = pd.Series(row["occupied"])

        # Limiter affichage pour la lisibilité
        if max_points is not None and len(lat) > max_points:
            latitude = latitude.iloc[:max_points]
            longitude = longitude.iloc[:max_points]
            occupied = occupied[:max_points]

        # Calcul fenêtre avec marge pour joli affichage
        max_lat = latitude.max()
        min_lat = latitude.min()
        max_lon = longitude.max()
        min_lon = longitude.min()

        marge_latitude = (max_lat - min_lat) * 0.05
        marge_longitude = (max_lon - min_lon) * 0.05

        window = dict(
            west=min_lon - marge_longitude,
            east=max_lon + marge_longitude,
            south=min_lat - marge_latitude,
            north=max_lat + marge_latitude,
        )
        center = {"lat": (window["south"] + window["north"]) / 2, "lon": (window["west"] + window["east"]) / 2}
        zoom = estimate_zoom_from_window(window)

        # Image    
        suffix = ""

        if occ_value is not None:
            if occ_value == 0:
                occ_text = "No client"
            else:
                occ_text = "With client"
            suffix += f" — {occ_text}"

        if trip_id is not None:
            suffix += f" — trip_id={trip_id}"


        fig = px.scatter_map(lat=latitude, 
                             lon=longitude, 
                             map_style='open-street-map', 
                             center=center, zoom=zoom, 
                             title=f"Trajet du taxi {cab_id}{suffix}, (points affichés: {len(latitude)})", 
        )

        if len(fig.data) > 0:
            fig.data[0].name = "Points gps"
            fig.data[0].showlegend = True

        return fig
    return (make_trip_map_from_seq_simple,)


@app.cell
def _(cab_seq_df, example_cab_id, make_trip_map_from_seq_simple):
    fig_test_simple = make_trip_map_from_seq_simple(cab_seq_df, example_cab_id)
    fig_test_simple
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous avons donc deux fonctions :
    - Pour afficher les trajets sous forme de ligne à partir d'un dataframe
    - Pour afficher juste les points : avec possibilité de zoomer.

    Ces fonctions prennent en entrée un dataframe.

    ## Découpage des données par trajets

    Puisqu'il s'avère complexe d'observer par taxi, le plus simple est de choisir un trajet pour un taxi pour l'afficher.
    Mais ne sachant pas encore quelles séries nous voulons considérer, nous allons entretenr 2 dataframe
    - Par taxi
    - Par trajets

    Les clustering seront alors testé sur chacun de ces datasets.
    Noius pouvons alors construire un second dataset à partir du premier.
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
    return (voyages_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Impression des courses :
    Nous pouvons afficher les courses pour observer ce type de trajets (plus courts)
    """)
    return


@app.cell
def _(
    cab_seq_df,
    make_trip_map_from_seq,
    make_trip_map_from_seq_simple,
    voyages_df,
):
    example_cab_id_2 = cab_seq_df["cab_id"].iloc[3]
    example_trip_id_2 = voyages_df.loc[voyages_df["cab_id"] == example_cab_id_2, "trip_id"].iloc[0]

    _fig_trajet_simple = make_trip_map_from_seq_simple(voyages_df, example_cab_id_2, trip_id=example_trip_id_2)
    _fig_trajet = make_trip_map_from_seq(voyages_df, example_cab_id_2, trip_id=example_trip_id_2)

    _fig_trajet_simple, _fig_trajet
    return


@app.cell
def _(cab_seq_df, mo):
    taxi_select = mo.ui.dropdown(
        options=sorted(cab_seq_df["cab_id"].unique().tolist()),
        value=cab_seq_df["cab_id"].iloc[0],
        label="Choisir un taxi"
    )

    taxi_select
    return (taxi_select,)


@app.cell
def _(mo, taxi_select):
    _selected_taxi_id = taxi_select.value
    mo.md(f"Taxi sélectionné : **{_selected_taxi_id}**")
    return


@app.cell
def _(mo, taxi_select, voyages_df):
    highest_trip_id = (voyages_df.loc[voyages_df["cab_id"] == taxi_select.value, "trip_id"].max())
    lowest_trip_id = (voyages_df.loc[voyages_df["cab_id"] == taxi_select.value, "trip_id"].min())
    slider_trip_id = mo.ui.slider(start=lowest_trip_id, stop=highest_trip_id, step=1)


    mo.md(
        f"""
        Choisir le voyage (selon l'id): {slider_trip_id}
        """
    )
    return (slider_trip_id,)


@app.cell
def _(
    make_trip_map_from_seq,
    make_trip_map_from_seq_simple,
    slider_trip_id,
    taxi_select,
    voyages_df,
):
    _fig_trajet_simple = make_trip_map_from_seq_simple(voyages_df, taxi_select.value, trip_id=slider_trip_id.value)
    _fig_trajet = make_trip_map_from_seq(voyages_df, taxi_select.value, trip_id=slider_trip_id.value)

    _fig_trajet_simple, _fig_trajet
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Trajets similaires

    Nous pouvons alors chercher des trajets similaires manuellement. Ceux-ci vont nous servir de témoin et tests pour évaluer les clusterings.

    Utilisons les outils précédents afin d'évaluer visuellement quels trajets se ressemblent. Voici une liste de types de trajets que nous voulons par exemple regrouper. Il s'agit de trajets suivant les mêmes routes / mêmes zones, mais aussi les mêmes types de trajets bien que leur localisations soient très distinctes. Cette liste servira à analyser les clusters obtenus.
    - Trajets d'attente d'un pour trouver un passager : Fais un petit trajet le temps de trouver un client. _Exemple :_ abnovkak 10920, eoffrij 308863, equioc 327096, ijcoga 454158. De plus, ubjaju 757651 sa recherche est très loin des autres (aéroport) mais suit un pattern similaire.
    - Déposer un client -> le long du Golden Gate park. _Exemples :_ iblool 398194, ijcoga 455363 (mais trajet plus long), occeyd 578423 et 578110 (comme ijcoga), unwrain 814508.
    - Trajet de l'aéroport de San Bruno (south San Francisco) - au centre ville (Civic Center) en passant par l'US 101 (grande voie rapide). _Examples :_ iblool 398959 et 398998, ibwicim 402633 et 402733.
    Mais attention, ibwicim 403134 trajet par la US 101 mais ne vient pas de l'aéroport ! Il conviendra de vérifier si ce trajet est mis dans le même cluster. De plus, on peut observer si occeyd 577695 qui fais le même trajet mais dans le sens inverse sans client va être dans le même cluster.
    - De Mission District à Union market street (centre) : _Examples :_ ubjaju 757539 (no client), unwrain 814517 (with client) et 814599 (même trajet mais la route parrallèle).

    Le cellules suivantes ont pour objectif de vérifier ces demandes.
    """)
    return


@app.cell
def _():
    GROUPS_DICT = {
        "Recherche passager (petits trajets d'attente)": [
            ("abnovkak", 10920),
            ("eoffrij", 308863),
            ("equioc", 327096),
            ("ijcoga", 454158),
            ("ubjaju", 757651),  # pattern similaire mais loin (aéroport)
        ],
        "Déposer client -> le long de Golden Gate Park": [
            ("iblool", 398194),
            ("ijcoga", 455363),  # plus long
            ("occeyd", 578423),
            ("occeyd", 578110),
            ("unwrain", 814508),
        ],
        "Aéroport San Bruno -> Civic Center via US 101": [
            ("iblool", 398959),
            ("iblool", 398998),
            ("ibwicim", 402633),
            ("ibwicim", 402733),
            ("ibwicim", 403134),  # via US101 mais pas aéroport (cas test)
            ("occeyd", 577695),   # sens inverse, no client (cas test)
        ],
        "Mission District -> Union / Market St": [
            ("ubjaju", 757539),   # no client
            ("unwrain", 814517),  # with client
            ("unwrain", 814599),  # route parallèle
        ],
    }
    return (GROUPS_DICT,)


@app.cell
def _(pd):
    def evaluate_cluster_groups(df, cluster_col: str, groups: dict, cab_col: str = "cab_id", trip_col: str = "trip_id",) -> dict:
        """
        Vérifie présence + clusters pour une liste de (cab_id, trip_id) regroupés.
        Affiche, par groupe témoin :
          - éléments présents/absents
          - cluster d'appartenance
          - verdict: "tous même cluster ?" (sur les présents)
        Retourne un dict de DataFrames (un par groupe) pour inspection.
        """
        if cab_col not in df.columns or trip_col not in df.columns:
            raise ValueError(f"Le dataframe doit contenir '{cab_col}' et '{trip_col}'.")
        if cluster_col not in df.columns:
            raise ValueError(f"Colonne de cluster '{cluster_col}' introuvable dans df.")

        out = {}

        for group_name, pairs in groups.items():
            rows = []
            for cab_id, trip_id in pairs:
                hit = df.loc[(df[cab_col] == cab_id) & (df[trip_col] == trip_id)]
                if hit.empty:
                    rows.append({
                        "group": group_name,
                        "cab_id": cab_id,
                        "trip_id": trip_id,
                        "present": False,
                        cluster_col: None
                    })
                else:
                    # si doublons improbables: on prend le 1er
                    cluster_val = hit.iloc[0][cluster_col]
                    rows.append({
                        "group": group_name,
                        "cab_id": cab_id,
                        "trip_id": trip_id,
                        "present": True,
                        cluster_col: cluster_val
                    })

            rep = pd.DataFrame(rows)

            present = rep[rep["present"]]
            clusters = present[cluster_col].dropna().unique().tolist()

            if len(present) == 0:
                verdict = "Aucun élément présent dans le dataset."
            elif len(clusters) <= 1:
                verdict = f"OK: tous les présents sont dans le même cluster ({clusters[0] if clusters else None})."
            else:
                verdict = f"NON: les présents sont répartis sur {len(clusters)} clusters: {clusters}"

            rep_sorted = rep.copy()
            rep_sorted["_cluster_sort"] = rep_sorted[cluster_col].astype("string")
            rep_sorted = rep_sorted.sort_values(["present", "_cluster_sort", "cab_id", "trip_id"], ascending=[False, True, True, True])
            rep_sorted = rep_sorted.drop(columns=["_cluster_sort"])

            print("\n" + "="*80)
            print(group_name)
            print("-"*80)
            print(verdict)
            print(rep_sorted.to_string(index=False))

            out[group_name] = rep_sorted

        return out
    return (evaluate_cluster_groups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering

    Le clustering de séries temporelles suit globalement une pipeline en trois étapes : représenter les séries, définir une mesure de dissimilarité entre elles, puis appliquer un algorithme de clustering pour regrouper les séries similaires.

    Les méthodes de clustering de séries temporelles se divisent en quatre grandes familles : distance-based, distribution-based, subsequence-based et representation-learning-based, chacune se distinguant par la manière dont elle compare ou représente les séries pour le clustering.

    L'analyse de la pipeline de clustering et de la taxonomie des différentes méthodes permet de construire le plan de route suivant :
    - Première analyse : Distance-based ou Distribution-based (feature-based)
      Utilisation de méthodes simples et classiques permettant d'obtenir un premier jet.
    - Recherche exploratoire motifs : Subsequence-based
    - Essaies avancés : Representation-learning
    L'analyse primaire permettra aussi de comparer des approches issues du cas univarié adaptées à nos séries multivariées GPS. Puis, nous nous pencherons plus spécifiquement sur le cas multivarié.

    ## Clustering classique

    ### Distance Based

    Commençons par étudier des méthode dites "distance-based". Nous allons appliquer certaines des méthodes du survey pouvant convenir à notre cas. Nous allons alors utiliser des méthodes reposant sur les données brutes.

    Selon l'article, pour s'appliquer au cas mutlivarié, il faut intégrer tous les canaux dans la distance. C'est dans cette démarche que nous allons appliquer les algorithmes suivants. Leurs fonctionnements sont similaires à k-means mais se distinguent pas les métriques utilisées notamment pour permettre leur application au cas multi-varié. Le "m" dans leur nom correspond d'ailleurs à cette adaptation.
    - m-kAVG + m-ED (euclidien distance) + k-Means
    - m-kDBA + m-DTW

    Une complexité liée à ce problème est la différence de taille des séquences. En effet, nous avons ici des séquences de tailles variables. Il convient alors de traiter les données.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Resample

    Pour les deux première pipelines, nous allons utiliser tslearn. Ce framework utilise les timeseries sous forme de tenseur, ce pourquoli les données sont ici adaptées. Le format attendu ici est X.shape = (n_samples, time_length, n_features). Sachant que time_length doit alors être constant ici et n_feature=2 dans ce cas (latitude, longitude).

    Nous allons d'abord utiliser l'option pour re sampler les données afin d'avoir des longueurs comparables. Ce sera utile pour la pipeline m-kAVG + m-ED. Nous pouvons alors sélectionner aléatoirement un certain nombre de séries. On y ajoute les séries que nous avons sélectionné pour réaliser nos tests.

    Le resampling consiste à choisir "T" points, avec ici arbitrairement 15 points puisque cela correspond environ à la taille des séries (allant de 3 à 30 points en général). Si la série en entrée a moins de points, la fonction en rajoute entre le points. Si la série en a plus, on obtient T points régulièrement espacés le long de la trajectoire originale, même si celle-ci avait beaucoup plus de points.

    On peut relever que l'échantillonnage réalisé ici correspond à une connaissance métier : nous avons découper selon ce que l'on sait être un trajet puis avons réechantillionné afin d'égaler la longueur des séries. Il faut alors avoir conscience de cette étape pour l'application des pipelines à d'autres jeux de données. D'autres moyens auraient pu être employés (sliding window etc.) mais correspondent ici moins au jeu de données. Dans tous les cas, ces méthodes se confronteraient au même problème de l'explosion du nombre de données.
    """)
    return


@app.cell
def _(np):
    def resample_2d(lat, lon, T):
        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)

        mask = np.isfinite(lat) & np.isfinite(lon)
        lat = lat[mask]
        lon = lon[mask]

        if lat.size < 2:
            return None

        x_old = np.linspace(0.0, 1.0, lat.size, dtype=np.float32) # Autant de points que la série en entrée
        x_new = np.linspace(0.0, 1.0, T, dtype=np.float32)        # T points comme voulu sur le même espace normalisé

        # Séries de T points : interpolation préservant la forme générale de la trajectoire
        lat_new = np.interp(x_new, x_old, lat).astype(np.float32)
        lon_new = np.interp(x_new, x_old, lon).astype(np.float32)

        return np.stack([lat_new, lon_new], axis=1)  # (T, 2)
    return (resample_2d,)


@app.cell
def _(GROUPS_DICT, np, resample_2d, voyages_df):
    T = 15     # Longueur arbitraire des séries (peut nécessiter d'ajouter ré-échantillonage)
    N = 15000  # Nombre de séries considérées dans le jeu de données
    N_big = 50000


    rng = np.random.default_rng(307)

    _df_work = voyages_df  

    _must_pairs = [(cab_id, trip_id) for pairs in GROUPS_DICT.values() for (cab_id, trip_id) in pairs]

    _must_idx = []
    _missing_pairs = []
    for cab, tid in _must_pairs:
        hit = _df_work.index[(_df_work["cab_id"] == cab) & (_df_work["trip_id"] == tid)]
        if len(hit) == 0:
            _missing_pairs.append((cab, tid))
        else:
            _must_idx.append(int(hit[0]))

    _must_idx = np.array(sorted(set(_must_idx)), dtype=int)

    _remaining = N - len(_must_idx)
    if _remaining > 0:
        all_idx = np.arange(len(_df_work), dtype=int)
        eligible = np.setdiff1d(all_idx, _must_idx, assume_unique=False)
        add_idx = rng.choice(eligible, size=min(_remaining, len(eligible)), replace=False)
        final_idx = np.concatenate([_must_idx, add_idx])
    else:
        final_idx = _must_idx[:N]

    # Même chose pour construire un second jeu d'indices plus grand
    _remaining_big = N_big - len(_must_idx)
    if _remaining_big > 0:
        all_idx = np.arange(len(_df_work), dtype=int)
        eligible = np.setdiff1d(all_idx, _must_idx, assume_unique=False)
        add_idx_big = rng.choice(eligible, size=min(_remaining_big, len(eligible)), replace=False)
        final_idx_big = np.concatenate([_must_idx, add_idx_big])
    else:
        final_idx_big = _must_idx[:N_big]

    # Construction jeu de données resamplé (petit)
    _X_list = []
    kept_rows = []
    for i in final_idx:
        row = _df_work.iloc[int(i)]
        ts = resample_2d(row["lat"], row["lon"], T)
        if ts is not None:
            _X_list.append(ts)
            kept_rows.append(int(i))


    X_resampled = np.stack(_X_list, axis=0)  
    df_subset = _df_work.iloc[kept_rows].reset_index(drop=True)


    # Mêmes données sans le resample (true data)

    _X_subset_list = []
    for row in df_subset.itertuples(index=False):
        lat = np.asarray(row.lat, dtype=np.float32)
        lon = np.asarray(row.lon, dtype=np.float32)
        _X_subset_list.append(np.column_stack([lat, lon]))  

    X_subset = np.array(_X_subset_list, dtype=object)  # utilisation array car on peut pas stack (différentes shapes)


    # construire aussi X_resampled_big et df_subset_big 
    _X_list_big = []
    kept_rows_big = []
    for i in final_idx_big:
        row = _df_work.iloc[int(i)]
        ts = resample_2d(row["lat"], row["lon"], T)
        if ts is not None:
            _X_list_big.append(ts)
            kept_rows_big.append(int(i))

    X_resampled_big = np.stack(_X_list_big, axis=0)
    df_subset_big = _df_work.iloc[kept_rows_big].reset_index(drop=True)


    len(_missing_pairs), X_resampled.shape, len(X_subset), df_subset.shape
    return X_resampled, X_resampled_big, df_subset, df_subset_big, lat


@app.cell
def _(mo):
    mo.md(r"""
    Le script précédent permettait d'obtenir deux éléments:
    - df_subset :un DataFrame pandas où chaque ligne = un trajet : cela permet de conserver les données brut mais en ne gardant que les données utilisées en entraînement.
    - X_subset et X_resampled: Ce sont des tableaux numpy d’objets. Chaque élément est une matrice (longueur série, 2). Ils serviront aux entraînements des algorithmes.

    Maintenant que nous avons à disposition des séries de tailles constante : X_resampled (arbitrairement de longueur 15), nous voulons choisir le nombre de cluster idéal k pour l'algiorithme k-means. Réalisons la méthode du coude pour trouver ce paramètre :
    """)
    return


@app.cell
def _():
    from tslearn.clustering import TimeSeriesKMeans
    SEED = 307
    return SEED, TimeSeriesKMeans


@app.cell
def _(SEED, TimeSeriesKMeans, X_resampled, np, plt, tqdm):
    def elbow_kmeans(X,k_min=2,k_max=20,metric="euclidean",max_iter=40,n_init=2,random_state=SEED):
        k_list = list(range(k_min, k_max + 1))
        inertias = []

        for k in tqdm(k_list):
            model = TimeSeriesKMeans(
                n_clusters=k,
                metric=metric,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state,
            )
            model.fit(X)
            inertias.append(model.inertia_)  

        return np.array(k_list), np.array(inertias)


    _k_list, _inertias = elbow_kmeans(
        X_resampled,
        k_min=2,
        k_max=16,
        metric="euclidean",
        max_iter=40,
        n_init=2,
        random_state=SEED,
    )


    plt.figure()
    plt.plot(_k_list, _inertias, marker="o")
    plt.xlabel("nombre de clusters : k")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude — pipeline (euclidean / mkAVG)")
    plt.xticks(_k_list)
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    On peut alors observer sur le graphe que le nombre idéal de cluster pour cet algorithme est 6. Nous pouvons alors utiliser ce paramètre afin d'observer qualitativement les clusters. En effet, après k=6, chaque cluster supplémentaire n’explique que très peu de variance alors que l'ajout de nouveaux clusters lors k<6 améliorait significativement la mesure d'inertie. Nous utiliserons ainsi pour la suite 6 clusters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### m-kAVG

    Nous pouvons alors visualiser et comparer une série après le resample. Puis nous pouvons appliquer le k-means sur ces séries.
    """)
    return


@app.cell
def _(TimeSeriesKMeans, X_resampled, df_subset):
    model = TimeSeriesKMeans(
        n_clusters=6,
        metric="euclidean",
        max_iter=50,
        random_state=307,
    )

    labels = model.fit_predict(X_resampled)

    voyages_clustered_subset = df_subset.copy()
    voyages_clustered_subset["cluster_mkAVG"] = labels
    voyages_clustered_subset.head()
    return (voyages_clustered_subset,)


@app.cell
def _(GROUPS_DICT, evaluate_cluster_groups, voyages_clustered_subset):
    reports = evaluate_cluster_groups(df=voyages_clustered_subset, cluster_col="cluster_mkAVG", groups=GROUPS_DICT)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Passage à DTW
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous avons forcé le fait de faire apparaître nos séries d'entraînement dans le jeu de test. Nous pouvons alors observer les résultats obtenus.
    - La première observation concerne la durée d'entraînement. Cette opération peut s'avérer particulièrement longue. En effet, k-means est très long a exécuter. C'est d'ailleurs pour cette raison que le jeu d'entraînement est ici restreint.
    - K-means nécessite de réaliser plusieurs entraînements afin de trouver le bon nombre de clusters. Cette méthode est spécifique au jeu de données, il convient alors de faire des tests pour son application.
    - Le fonctionnement appliqué au cas multivarié (ici d=2) correspond ici à la somme des distance euclidienne sur chaque dimension. Nous avons donc une importance égale pour chaque dimension. Ce n'est pas un problème pour ce jeu de données mais peut s'avérer être un problème pour d'autres.
    - L'exploration qualitative permet de constater que lorsque nous avons 6 clusters, certains groupes que nous voulons ensemble se retrouvent tous ensemble. C'est le cas des trajets autour du golden gate park, les trajets partant de l'aéroport se retrouvent ensemble
    - Mais qu'il y a des exceptions ...
    - Comme attendu, le groupe des petits trajets se retrouvent dans des catégories différentes car à des endroits variés. cependant, certains de ces trajets sont similaires et dans les mêmes zones et se retrouvent donc ensemble.

    Globalement, on peut s'intérroger sur le choix de découper nos trajets en 6 clusters : est-ce que nous avons en effet seulement 6 trajets caractéristiques ?

    ### m-kDBA + m-DTW

    Nous cherchons maintenant à utiliser le DTW. Cette méthode permet d'entrer des séries de tailles différentes ! Cela peut-être particulièrement pratique dans le cas des trajets, leurs tailles peuvent fortement varier. testons si avec les différence de taille cela peut fonctionner.

    On peut cependant relever que le calcul de DTW est couteux et cela peut s'avérer long. C'est ce que nous allons observer pour la suite. Commençons par définir la pipeline du model :
    """)
    return


@app.cell
def _(
    SEED,
    TimeSeriesKMeans,
    X_resampled_big,
    df_subset_big,
    voyages_clustered_subset_dtw,
):
    model_dtw = TimeSeriesKMeans(
        n_clusters=8,            # A adapter avec la méthode du coude
        metric="dtw",            # distance DTW utilisée. doc : If “dtw”, DBA is used for barycenter computation.
        max_iter=40,
        random_state=SEED,       # Défini à 307 plus tôt
    )

    labels_dtw_big = model_dtw.fit_predict(X_resampled_big)

    # Pour le moment : copie de df_subset mais plus tard fusionner toutes les observations pour comparer
    voyages_clustered_subset_dtw_big = df_subset_big.copy()
    voyages_clustered_subset_dtw_big["cluster_mkDBA"] = labels_dtw_big
    voyages_clustered_subset_dtw.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nous pouvons alors observer les résultats obtenus sur notre test qualitatif :
    """)
    return


@app.cell
def _(GROUPS_DICT, evaluate_cluster_groups, voyages_clustered_subset_dtw):
    reports_dtw = evaluate_cluster_groups(df=voyages_clustered_subset_dtw, cluster_col="cluster_mkDBA", groups=GROUPS_DICT)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
