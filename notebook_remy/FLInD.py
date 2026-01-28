import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="FLInD", css_file="custom.css")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import marimo as mo

    import csv
    import gc
    import glob
    from io import TextIOWrapper

    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import mpl_toolkits.mplot3d.art3d as art3d

    import networkx as nx
    import numpy as np

    # Stats
    from numpy import mean, var, std

    from pathlib import Path
    import polars as pl
    import psycopg2
    from scipy import stats
    import seaborn as sns
    import sys
    import taosrest
    import tarfile
    import tempfile
    import time
    from typing import Union
    import requests
    import urllib.request
    import zipfile


@app.function(hide_code=True)
# Convenience method to import citations from bibtex lib into the notebook
def cite(ref_name: str) -> str:
    """
    Returns human-readable reference matching input `ref_name` value.

    This method imports citations from a bibtex library file (.bib),
    allowing to share the same references between the research paper and
    this very notebook.

    Original method from exploration notebook relies on `bibtexparser`
    package, but we relax this dependency here to ease deployment of this
    very notebook.
    """

    with open('references.bib') as bibtex_file:
        entries_dict = {}
        current_entry_name = ''

        lines = [line.rstrip() for line in bibtex_file]
        for line in lines:
            has_opening_bracket = '{' in line
            has_closing_bracket = '}' in line

            # not of interest for parsing
            if not has_opening_bracket and not has_closing_bracket:
                continue

            # new entry
            if has_opening_bracket and not has_closing_bracket:
                name = line[line.index('{')+1:-1]
                entries_dict[name] = {}
                current_entry_name = name
                continue

            # load info
            if 'author' in line.lower():
                f_index = line.index('{')
                l_index = len(line) - 1 - line[::-1].index('}')
                entries_dict[current_entry_name]['author'] = line[f_index+1:l_index]
                continue
            if 'year' in line.lower():
                f_index = line.index('{')
                l_index = len(line) - 1 - line[::-1].index('}')
                entries_dict[current_entry_name]['year'] = line[f_index+1:l_index]
                continue
            if 'url =' in line.lower() or 'url=' in line.lower():
                f_index = line.index('{')
                l_index = len(line) - 1 - line[::-1].index('}')
                entries_dict[current_entry_name]['url'] = line[f_index+1:l_index]
                continue

        if ref_name not in entries_dict:
            raise NameError(f"Entry '{ref_name}' not found in references.bib.")
        entry = entries_dict[ref_name]

        return f"[[{entry['author']}, {entry['year']}]({entry['url']})]"


@app.function(hide_code=True)
# Print with time
def dprint(content):
    """
    Prints {content} value to standard input, with human-readable time
    value as a prefix.
    """

    s=time.gmtime(time.time())
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", s)}] {content}')


@app.function(hide_code=True)
# Reset Matplotlib font scale
def reset_plt_font_scale():
    """
    Resets Matplotlib.plt font size to its default value.

    This is needed since font size value is reduced when drawing heatmap
    plots, and font size can only be modified globally (as far as I know
    right now).
    """

    plt.rcParams.update({'font.size': 10})


@app.class_definition(hide_code=True)
# Fast linear interpolation Python implementation
class FastLinearInterpolation:
    """
    Python implementation of the FLI model.

    Rémy Raes, Olivier Ruas, Adrien Luxey-Bitri, Romain Rouvoy. Compact
    Storage of Data Streams in Mobile Devices. DAIS'24 - 24th International
    Conference on Distributed Applications and Interoperable Systems, Jun
    2024, Groningen, Netherlands. pp.45. ⟨hal-04969154v3⟩
    """

    def __init__(self):
        self.firstStartingTimeRaw: float = -1
        self.listOldModels = []
        self.currentStartingTimeRaw: float = 0
        self.currentStartingValueRaw: float = 0
        self.currentA: float = 0
        self.maxPossibleA: float = sys.float_info.max
        self.minPossibleA: float = sys.float_info.min
        self.lastTimeRaw: float = 0
        self.lastValueRaw: float = 0
        self.maxError: float = 0
        self.pointsCount: int = 0
        self.segmentsCount: int = 0

    def setError(self, value: float) -> None:
        self.maxError = value

    def add(self, tRaw: float, vRaw: float):
        if self.firstStartingTimeRaw == -1:
            self.firstStartingTimeRaw = tRaw
            self._initModel(tRaw, vRaw)
            return
        tNorm: float = tRaw - self.currentStartingTimeRaw
        vNorm: float = vRaw - self.currentStartingValueRaw
        A: float = vNorm / tNorm
        if A <= self.maxPossibleA and A >= self.minPossibleA:
            self.minPossibleA = max(self.minPossibleA, (vNorm - self.maxError) / tNorm)
            self.maxPossibleA = min(self.maxPossibleA, (vNorm + self.maxError) / tNorm)
            self.lastTimeRaw = tRaw
            self.lastValueRaw = vRaw
            self.currentA = A
            self.pointsCount += 1
            return
        self.listOldModels.append((self.currentStartingTimeRaw, self.currentStartingValueRaw, self.currentA, self.pointsCount, self.segmentsCount))
        self.currentStartingTimeRaw = self.lastTimeRaw
        self.currentStartingValueRaw = self.lastValueRaw
        tNorm = tRaw - self.lastTimeRaw
        vNorm = vRaw - self.lastValueRaw
        A = vNorm / tNorm
        self.pointsCount = 1
        self.segmentsCount += 1
        self.minPossibleA = (vNorm - self.maxError) / tNorm
        self.maxPossibleA = (vNorm + self.maxError) / tNorm
        self.lastTimeRaw = tRaw
        self.lastValueRaw = vRaw
        self.currentA = A

    def _initModel(self, tRaw: float, vRaw: float) -> None:
        self.currentStartingTimeRaw = tRaw
        self.currentStartingValueRaw = vRaw
        self.currentA = 0
        self.maxPossibleA = sys.float_info.max
        self.minPossibleA = sys.float_info.min
        self.lastTimeRaw = tRaw
        self.lastValueRaw = vRaw
        self.pointsCount = 1
        self.segmentsCount = 1

    def read(self, t: float) -> float:
        if self.firstStartingTimeRaw == -1 or t < self.firstStartingTimeRaw:
            raise IndexError('Error the time is unavailable')
        if self.currentStartingTimeRaw <= t:
            return self._getValue(t, self.currentStartingTimeRaw, self.currentA, self.currentStartingValueRaw)
        if len(self.listOldModels) == 0:
            raise IndexError('Error the time is unavailable (!)')
        _index: int = self._getIndexRead(t)
        model = self.listOldModels[_index]
        return self._getValue(t, model[0], model[2], model[1])

    def _getIndexRead(self, t: float) -> int:
        minIndex: int = 0
        maxIndex: int = len(self.listOldModels) - 1
        _index: int = -1
        tIndex: float
        while minIndex <= maxIndex:
            _index = (maxIndex + minIndex) // 2
            if minIndex == maxIndex:
                break
            tIndex = self.listOldModels[_index][0]
            if t == tIndex:
                break
            if t > tIndex and t < self.listOldModels[_index + 1][0]:
                break
            if t < tIndex:
                maxIndex = _index - 1
            else:
                minIndex = _index + 1
        return _index

    def _getValue(self, t: float, T: float, A: float, B: float) -> float:
        return A * (t - T) + B

    def data(self):
        points = list(map(lambda t: (t[0], t[1]), self.listOldModels))
        points.append((self.currentStartingTimeRaw, self.currentStartingValueRaw))
        points.append((self.lastTimeRaw, self.lastValueRaw, self.currentA))
        return points

    def get_modeled_points_counts(self):
        points = list(map(lambda t: (t[0], t[3]), self.listOldModels))
        points.append((self.currentStartingTimeRaw, self.pointsCount))
        points.append((self.lastTimeRaw, self.pointsCount))
        return points

    def get_segments_counts(self):
        points = list(map(lambda t: (t[0], t[4]), self.listOldModels))
        points.append((self.currentStartingTimeRaw, self.segmentsCount))
        points.append((self.lastTimeRaw, self.segmentsCount))
        return points

    def size(self):
        """
        Simulates models only storing time and value information (without slope information,
        as described in the paper).
        """

        return 2 * len(self.listOldModels) * 8


@app.function(hide_code=True)
def can_merge(tolerated_error: float, p1, p2, nxp) -> bool:
    """
    Tells whether a point can be forgotten while inserting data in a FLI
    instance.

    This computes the slope between `p1` and `nxp` unidimensional
    (timestamp, value) points, computes the projected value of `p2` with
    this slope, and compares it to the input `tolerated_error` value.
    """

    a: float = (nxp[1] - p1[1]) / (nxp[0] - p1[0])
    value: float = abs(a * (p2[0] - p1[0]) + p1[1] - p2[1])
    return value <= tolerated_error


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # Introduction

    ## Fast linear interpolation

    * Linear modelling of time series
    * `epsilon` tolerated error
    """)
    return


@app.cell(hide_code=True)
def _():
    error_slider = mo.ui.slider(0, 3, 0.1, debounce=True)
    return (error_slider,)


@app.cell(hide_code=True)
def _(error_slider):
    mo.md(f"""
    #### Core hint of the algorithm

    During runtime, upper and lower bounds decide whether to accept new incoming points or spawning a new model.

    In this figure, we check if two segments [AB] and [BC] can be merged.
    Two segments can be merged if the projection of the middle point (here B) on the potential new segment fits a given tolerated error.

    Tolerated error: {error_slider}
    """)
    return


@app.cell(hide_code=True)
def _(error_slider):
    def _():
        reset_plt_font_scale()
        tolerated_error = error_slider.value
        plt.figure(figsize=(8, 4))
        plt.axis([0, 10, 0, 7])
        p1 = (1, 1)
        p2 = (5, 2)
        p3 = (9, 6)
        plt.annotate('A', (p1[0] - 0.25, p1[1]))
        plt.annotate('B', (p2[0] + 0.12, p2[1] - 0.2))
        plt.annotate('C', (p3[0], p3[1] + 0.15))
        plt.plot([-1, p1[0]], [-1, p1[1]], marker='o', color='black')
        plt.plot([p3[0], 11], [p3[1], 7], marker='o', color='black')
        mergeable: bool = can_merge(tolerated_error, p1, p2, p3)
        plt.text(5, 0.4, 'can_merge(error={error}, A, B, C) = {result}'.format(error=round(tolerated_error, 2), result=mergeable), horizontalalignment='center', verticalalignment='center')
        point_color = 'green' if mergeable else 'red'
        poly_color = '#0f05' if mergeable else '#f005'
        plt.plot([p1[0], p3[0]], [p1[1], p3[1]], linestyle='dashed', linewidth=1, color='grey')
        slope = (p3[1] - p1[1]) / (p3[0] - p1[0])
        b = (p3[0] * p1[1] - p1[0] * p3[1]) / (p3[0] - p1[0])
        p_candidate = (p2[0], slope * p2[0] + b)
        plt.plot(p_candidate[0], p_candidate[1], 'go' if mergeable else 'ro', markersize=3)
        p_candidate_bottom = (p_candidate[0], p_candidate[1] - tolerated_error)
        p_candidate_top = (p_candidate[0], p_candidate[1] + tolerated_error)
        slope = (p_candidate_bottom[1] - p1[1]) / (p_candidate_bottom[0] - p1[0])
        b = (p_candidate_bottom[0] * p1[1] - p1[0] * p_candidate_bottom[1]) / (p_candidate_bottom[0] - p1[0])
        c_bottom = (p3[0], slope * p3[0] + b)
        plt.plot([p1[0], c_bottom[0]], [p1[1], c_bottom[1]], linestyle='dotted', linewidth=1, color=point_color)
        slope = (p_candidate_top[1] - p1[1]) / (p_candidate_top[0] - p1[0])
        b = (p_candidate_top[0] * p1[1] - p1[0] * p_candidate_top[1]) / (p_candidate_top[0] - p1[0])
        c_top = (p3[0], slope * p3[0] + b)
        plt.plot([p1[0], c_top[0]], [p1[1], c_top[1]], linestyle='dotted', linewidth=1, color=point_color)
        plt.quiver(p_candidate[0], p_candidate[1], 0, tolerated_error, angles='xy', scale_units='xy', scale=1, width=0.003, headwidth=5, color=point_color)
        plt.text(p_candidate[0] + 0.05, p_candidate[1] + tolerated_error / 2, 'err={e}'.format(e=round(tolerated_error, 1)), color=point_color)
        plt.quiver(p_candidate[0], p_candidate[1], 0, -1 * tolerated_error, angles='xy', scale_units='xy', scale=1, width=0.003, headwidth=5, color=point_color)
        plt.text(p_candidate[0] - 1.15, p_candidate[1] - tolerated_error / 2, 'err={e}'.format(e=round(tolerated_error, 1)), color=point_color)
        points = [p1, c_bottom, c_top, p1]
        xs, ys = zip(*points)
        plt.fill(xs, ys, poly_color)
        points = [p1, p2, p3]
        timestamps, values = zip(*points)
        plt.plot(timestamps, values, marker='o', color='black')
        return plt.gca()

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Toy example
    """)
    return


@app.cell(hide_code=True)
def _():
    toy_2d_error_slider = mo.ui.slider(0, 3, 0.1, label='Tolerated error:', show_value=True, debounce=True)
    toy_2d_iteration_slider = mo.ui.slider(0, 5, 1, label='Iteration:', show_value=True, debounce=True)
    return toy_2d_error_slider, toy_2d_iteration_slider


@app.cell(hide_code=True)
def _(toy_2d_error_slider, toy_2d_iteration_slider):
    def _():
        tolerated_error = toy_2d_error_slider.value
        iteration = toy_2d_iteration_slider.value
        plt.figure(figsize=(8, 4))
        p1 = (1727860678221, 10)
        p2 = (1727860678225, 12)
        p3 = (1727860678229, 16)
        p4 = (1727860678231, 16)
        p5 = (1727860678235, 18)
        p6 = (1727860678236, 19)
        plt.annotate('A', (p1[0] - 0.25, p1[1]))
        plt.annotate('B', (p2[0] + 0.12, p2[1] - 0.2))
        plt.annotate('C', (p3[0], p3[1] + 0.15))
        plt.annotate('D', (p4[0], p4[1] + 0.2))
        plt.annotate('E', (p5[0], p5[1] + 0.2))
        plt.annotate('F', (p6[0], p6[1] + 0.2))
        points = [p1, p2, p3, p4, p5, p6]
        timestamps, values = zip(*points)
        plt.plot(timestamps, values, marker='o', color='black', label='Original data')
        model = FastLinearInterpolation()
        model.setError(tolerated_error)
        compressed = []
        for i in range(0, iteration + 1):
            model.add(points[i][0], points[i][1])
            compressed = model.data()
        if tolerated_error > 0:
            for i in range(1, len(points)):
                p = points[i]
                plt.plot(p[0], p[1] - model.maxError, markersize=5, color='red', marker=6)
                plt.plot(p[0], p[1] + model.maxError, markersize=5, color='red', marker=7)
        c_timestamps, c_values = zip(*compressed)
        reset_plt_font_scale()
        plt.plot(c_timestamps, c_values, marker='o', color='orange', label='Compressed data')
        plt.title('Iteration #{i} of FLI process (tolerated error = {e})'.format(i=iteration, e=round(tolerated_error, 2)))
        plt.ylabel('Value')
        plt.xlabel('Time index')
        plt.axis([points[0][0] - 1, points[-1][0] + 1, points[0][1] - 1, points[-1][1] + 1])
        plt.legend()

        return mo.hstack([
            mo.vstack([
                toy_2d_error_slider,
                mo.md('The above slider controls the input error of FLI; increasing it reduces the count of preserved points and thus final size of the model, but also degrades data accuracy.<br/><br/>'),
                toy_2d_iteration_slider,
                mo.md('Increase the above cursor step by step to visualize how the model behaves when accepting new points.')
            ]),
            plt.gca()
        ])
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generalizing to 2 dimensions

    * Original `FLI` model made for univariate time series (UTS)
        * Relies on segments with the form $(timestamp, start\_value)$
    * To model MTS, we need to use several instances of `FLI`
        * In the INTACT paper, to model GPS data, two FLI instances are used (one for longitude, one for latitude)
    * Using several instances of `FLI` necessarily induces useless redundancy (the time dimension is captured several times, in each `FLI` instance)
    """)
    return


@app.cell(hide_code=True)
def _():
    toy_location_data = [
        (0, 3.11, 42.59),
        (1, 3.11, 42.58),
        (2, 3.12, 42.60),
        (3, 3.13, 42.59),
        (4, 3.15, 42.60),
        (5, 3.13, 42.58),
        (6, 3.15, 42.56),
        (7, 3.16, 42.57),
        (8, 3.17, 42.56),
        (10, 3.17, 42.57),
        (11, 3.162, 42.59)
    ]
    return (toy_location_data,)


@app.cell(hide_code=True)
def _(toy_location_data):
    def _():
        timestamps = [p[0] for p in toy_location_data]
        latitudes = [p[1] for p in toy_location_data]
        longitudes = [p[2] for p in toy_location_data]

        reset_plt_font_scale()
        plt.figure(figsize=(10, 3))

        # latitude
        plt.subplot(121)
        plt.plot(timestamps, latitudes, marker="o", color="green")
        plt.xlabel('Time')
        plt.ylabel('Latitude')
        plt.title('Latitude time series')

        # longitude
        plt.subplot(122)
        plt.plot(timestamps, longitudes, marker="o", color="blue")
        plt.xlabel('Time')
        plt.ylabel('Longitude')
        plt.title('Longitude time series')

        return plt.gca()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To avoid that redundancy, we propose to explore modelling GPS data with `FLI3D`, a `FLI` extension that captures GPS data by extending the segment format to cover both latitude and longitude dimensions: $(timestamp, lat\_start\_value, lng\_start\_value)$
    """)
    return


@app.cell(hide_code=True)
def _():
    synopsis_error_slider = mo.ui.slider(0, 0.02, 0.001, 0.01, debounce=True, label='Input error:', show_value=True)
    synopsis_roll_slider = mo.ui.slider(-90, 0, 5, -35, debounce=True, label='Roll controller:', show_value=True)
    return synopsis_error_slider, synopsis_roll_slider


@app.cell(hide_code=True)
def _(synopsis_error_slider, synopsis_roll_slider, toy_location_data):
    # Paper synopsis figure
    def _():
        timestamps = [p[0] for p in toy_location_data[:-1]]
        latitudes = [p[1] for p in toy_location_data[:-1]]
        longitudes = [p[2] for p in toy_location_data[:-1]]

        reset_plt_font_scale()
        plt.rcParams.update({'font.size': 9})
        plt.figure(figsize=(8, 4), dpi=200)

        # error
        d = synopsis_error_slider.value


        # latitude
        ax1 = plt.subplot(221)
        ## candidate
        p = toy_location_data[-3]
        l = toy_location_data[-2]
        c = toy_location_data[-1]

        # acceptance cone
        ## top break
        l_top_limit = [l[0], l[1]+d]
        slope = (l_top_limit[1] - p[1]) / (l_top_limit[0] - p[0])
        b = (l_top_limit[0] * p[1] - p[0] * l_top_limit[1]) / (l_top_limit[0] - p[0])
        p_candidate = (c[0], slope * c[0] + b)
        plt.plot(p_candidate[0], p_candidate[1], color='green', alpha=.4, marker="v")
        plt.plot([p[0], p_candidate[0]], [p[1], p_candidate[1]], color='green', alpha=.4, marker=None, linestyle='--')
        plt.quiver(l[0], l[1], 0, d, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=5, color='green') #vector
        ## bottom break
        l_top_limit = [l[0], l[1]-d]
        slope = (l_top_limit[1] - p[1]) / (l_top_limit[0] - p[0])
        b = (l_top_limit[0] * p[1] - p[0] * l_top_limit[1]) / (l_top_limit[0] - p[0])
        p_candidate_b = (c[0], slope * c[0] + b)
        plt.plot(p_candidate_b[0], p_candidate_b[1], color='green', alpha=.4, marker="^")
        plt.plot([p[0], p_candidate_b[0]], [p[1], p_candidate_b[1]], color='green', alpha=.4, marker=None, linestyle='--')
        plt.quiver(l[0], l[1], 0, -d, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=5, color='green') #vector
        ## cone
        points = [p, p_candidate, p_candidate_b, p]
        xs, ys = zip(*points)
        # Fill needs to be done twice for hatches to appear on output PDF files
        plt.fill(xs, ys, 'green', alpha=0.2, linewidth=0)
        plt.fill(xs, ys, color=None, fill=None, edgecolor="darkgreen", alpha=0.4, linewidth=0, hatch='///')
        ## series
        plt.plot(timestamps, latitudes, marker="o", color="green")
        plt.plot([p[0], c[0]], [p[1], c[1]], linestyle='dotted', color="green")
        plt.plot(c[0], c[1], marker="x", color='green')
        plt.xlabel('Time')
        plt.ylabel('Latitude')
        plt.title('Separate models')

        max_lat_value = p_candidate[1]
        min_lat_value = p_candidate_b[1]


        # longitude
        plt.subplot(223, sharex=ax1)
        ## candidate
        p = [toy_location_data[-3][0], toy_location_data[-3][2]]
        l = [toy_location_data[-2][0], toy_location_data[-2][2]]

        ## top break
        l_top_limit = [l[0], l[1]+d]
        slope = (l_top_limit[1] - p[1]) / (l_top_limit[0] - p[0])
        b = (l_top_limit[0] * p[1] - p[0] * l_top_limit[1]) / (l_top_limit[0] - p[0])
        p_candidate = (c[0], slope * c[0] + b)
        plt.plot(p_candidate[0], p_candidate[1], color='blue', alpha=.4, marker="v")
        plt.plot([p[0], p_candidate[0]], [p[1], p_candidate[1]], color='blue', alpha=.4, marker=None, linestyle='--')
        plt.quiver(l[0], l[1], 0, d, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=5, color='blue')
        ## bottom break
        l_top_limit = [l[0], l[1]-d]
        slope = (l_top_limit[1] - p[1]) / (l_top_limit[0] - p[0])
        b = (l_top_limit[0] * p[1] - p[0] * l_top_limit[1]) / (l_top_limit[0] - p[0])
        p_candidate_b = (c[0], slope * c[0] + b)
        plt.plot(p_candidate_b[0], p_candidate_b[1], color='blue', alpha=.4, marker="^")
        plt.plot([p[0], p_candidate_b[0]], [p[1], p_candidate_b[1]], color='blue', alpha=.4, marker=None, linestyle='--')
        plt.quiver(l[0], l[1], 0, -d, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=5, color='blue') #vector
        ## cone
        points = [p, p_candidate, p_candidate_b, p]
        xs, ys = zip(*points)
        plt.fill(xs, ys, 'blue', alpha=0.2, linewidth=0)
        plt.fill(xs, ys, color=None, fill=None, edgecolor="darkblue", alpha=0.4, linewidth=0, hatch='\\\\\\')
        ## Series
        plt.plot(timestamps, longitudes, marker="o", color="blue")
        plt.plot([p[0], c[0]], [p[1], c[2]], linestyle='dotted', color="blue")
        plt.plot(c[0], c[2], marker="x", color='blue')
        plt.xlabel('Time')
        plt.ylabel('Longitude')
        #plt.title('Longitude time series')

        max_lng_value = p_candidate[1]
        min_lng_value = p_candidate_b[1]


        # flind
        _ax = plt.subplot(1, 2, (2, 4), projection='3d')

        ## candidate
        min_lat = min([t[1] for t in toy_location_data]) - 0.05
        max_lat = max([t[1] for t in toy_location_data]) + 0.05
        min_lng = min([t[2] for t in toy_location_data]) - 0.05
        max_lng = max([t[2] for t in toy_location_data]) + 0.05
        max_time = max(t[0] for t in toy_location_data)

        # style settings
        band_opacity = 0.2
        projection_line_opacity = 0.8
        projection_marker_size = 10
        projection_line_width = 0.9

        _ax.set_xlim(min_lat, max_lat)
        _ax.set_ylim(0, max_time)
        _ax.set_zlim(min_lng, max_lng)

        ### latitude
        p = toy_location_data[-3]
        c = toy_location_data[-1]

        #### acceptance area
        #### needs to be done twice (https://stackoverflow.com/questions/5195466/matplotlib-does-not-display-hatching-when-rendering-to-pdf)
        rect = Rectangle((min_lat_value, min_lng), max_lat_value-min_lat_value, max_lng-min_lng, fill=True, color='green', zorder=0, alpha=band_opacity)
        _ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=max_time, zdir='y')
        rect = Rectangle((min_lat_value, min_lng), max_lat_value-min_lat_value, max_lng-min_lng, fill=None, color=None, zorder=0, alpha=band_opacity*2, hatch='///', edgecolor='darkgreen')
        _ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=max_time, zdir='y')

        #### dotted lines
        #_ax.plot([p[1], max_lat_value], [p[0], c[0]], zs=[p[2], c[2]], color='green', alpha=projection_line_opacity, linewidth=projection_line_width, linestyle='--', zorder=100)
        _ax.scatter(max_lat_value, c[0], s=projection_marker_size, zs=c[2], zdir='timestamp', color='green', marker="<", alpha=projection_line_opacity)
        #_ax.plot([p[1], min_lat_value], [p[0], c[0]], zs=[p[2], c[2]], color='green', alpha=projection_line_opacity, linewidth=projection_line_width, linestyle='--', zorder=100)
        _ax.scatter(min_lat_value, c[0], s=projection_marker_size, zs=c[2], zdir='timestamp', color='green', marker=">", alpha=projection_line_opacity)

        ### longitude
        #### acceptance area
        #### needs to be done twice (https://stackoverflow.com/questions/5195466/matplotlib-does-not-display-hatching-when-rendering-to-pdf)
        rect = Rectangle((min_lat, min_lng_value), max_lat - min_lat, max_lng_value-min_lng_value, fill=True, color='blue', zorder=0, alpha=band_opacity)
        _ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=max_time, zdir='y')
        rect = Rectangle((min_lat, min_lng_value), max_lat - min_lat, max_lng_value-min_lng_value, color=None, zorder=0, alpha=band_opacity, hatch='\\\\\\', edgecolor='blue')
        _ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=max_time, zdir='y')

        #### dotted lines
        #_ax.plot([p[1], c[1]], [p[0], c[0]], zs=[p[2], max_lng_value], color='blue', alpha=projection_line_opacity, linewidth=projection_line_width, linestyle='--', zorder=100)
        _ax.scatter(c[1], c[0], s=projection_marker_size, zs=max_lng_value, zdir='timestamp', color='blue', marker="v", alpha=projection_line_opacity)
        #_ax.plot([p[1], c[1]], [p[0], c[0]], zs=[p[2], min_lng_value], color='blue', alpha=projection_line_opacity, linewidth=projection_line_width, linestyle='--', zorder=100)
        _ax.scatter(c[1], c[0], s=projection_marker_size, zs=min_lng_value, zdir='timestamp', color='blue', marker="^", alpha=projection_line_opacity)

        ## series
        for _index, datum in enumerate(toy_location_data):
            timestamp = datum[0]
            latitude = datum[1]
            longitude = datum[2]
            is_last_point = _index == len(toy_location_data) - 1

            # marker
            _ax.scatter(latitude, timestamp, zs=longitude, zdir='timestamp', color='black', marker=('x' if is_last_point else 'o'))
            # line
            if _index == 0 or is_last_point:
                continue
            previous = toy_location_data[_index - 1]
            _ax.plot([previous[1], latitude], [previous[0], timestamp], zs=[previous[2], longitude], color='black', zorder=100, linestyle='-')

        ## potential new model
        l = toy_location_data[-3]
        c = toy_location_data[-1]
        _ax.plot([l[1], c[1]], [l[0], c[0]], zs=[l[2], c[2]], color='#444', zorder=200, linestyle='dotted')

        _ax.set_xlabel('Latitude')
        _ax.set_ylabel('Time')
        _ax.set_zlabel('Longitude')
        _ax.view_init(elev=20.0, azim=synopsis_roll_slider.value, roll=0)
        _ax.set_box_aspect(None, zoom=1)
        plt.title('Combined model')

        return mo.vstack([
            mo.hstack([synopsis_error_slider, synopsis_roll_slider]),
            #mo.md(f'max values for candidate point: lat:[{min_lat_value}, {max_lat_value}], lng:[{min_lng_value}, {max_lng_value}]'),
            plt.gca()
        ])
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # A model for storing multivariate time series
    """)
    return


@app.class_definition(hide_code=True)
# FLInD implementation
class FLInD(FastLinearInterpolation):
    """
    n-Dimensional extension of the FastLinearInterpolation class.

    This adds multivariate time series support to FLI, allowing to store
    series containing multiple values per timestamp:

        [timestamp, v_0, v_1, v_2, ..., v_n]

    Attributes
    ----------
    dimensions_count : int
        Number of dimensions accepted by the instance.
    segment_length : int
        Size of stored tuples, used internally to ensure {dimensions_count} is
        respected at all times.
    Methods
    -------
    name()
        Returns the name of the current instance.
    add(time, value)
        Adds a new data point.
    read(time)
        Returns values associated to input {time}.

    data()
        Returns preserved points.
    size()
        Returns the size of the instance, in bytes.
    complete_data()
        Returns preserved points, plus in-memory current point.
    complete_size()
        Returns size of {complete_data}, in bytes.

    add_frame(frame, series, sample_size):
        Convenience method loading the instance with a subpart of a dataframe.

    _getValue(time, slopes, startingValues)
        Internal method computing the data point associated to input {time}.
    _getIndexRead(time)
        Internal method returning the index of the point matching input {time}.
    """

    def __init__(self, dimensions_count, errors = []):
        self.dimensions_count = dimensions_count
        self.errors = errors
        self.segment_length = 1 + self.dimensions_count
        self.local_init()
        FastLinearInterpolation.__init__(self)

    def local_init(self):
        """
        Initializes the instance.

        This assigns as many zeroes as supported dimensions count to internal
        state.
        """

        self.currentStartingValues = np.empty(self.dimensions_count)
        self.currentValues = np.empty(self.dimensions_count)
        self.currentSlopes = np.empty(self.dimensions_count)
        self.minPossibleSlopes = np.full(self.dimensions_count, sys.float_info.min)
        self.maxPossibleSlopes = np.full(self.dimensions_count, sys.float_info.max)
        self.lastRawValues = np.empty(self.dimensions_count)

    def setError(self, value: float):
        if (len(self.errors) == 0):
            self.errors = [value for i in range(self.dimensions_count)]
        else:
            dprint("Errors array is already initialized, skipping.")

    def _initModel(self, tRaw: float, vRaw: tuple) -> None:
        """
        Sets current state using one data point.
        """

        self.currentStartingTimeRaw = tRaw
        self.lastTimeRaw = tRaw
        self.currentStartingValues = np.array(vRaw)
        self.lastRawValues = np.array(vRaw)

    def name(self):
        return f'FLInD(n={self.dimensions_count})'

    def data(self):
        return self.listOldModels

    def size(self):
        """
        Does not take in-memory params into account.
        """
        return self.segment_length * len(self.listOldModels) * 8 # 8 bytes per float

    def complete_data(self):
        currentModel = [self.currentStartingTimeRaw] + self.currentStartingValues
        return self.listOldModels + [currentModel]

    def complete_size(self):
        """
        Takes in-memory parameters into account.
        """
        return self.size() + self.segment_length * 8

    def add(self, tRaw: float, vRaw: tuple):
        """
        Adds a new data point.

        Input data point being splitted between its time and value components,
        this checks whether the new point fits the current interpolations; for
        this to be true, each value of the input point must fit associated
        interpolation.

        If all dimensions of the point fit, current interpolations are updated,
        reducing the acceptable data interval between upper and lower bounds
        for future points; if at least one dimension does not fit, current
        in-memory point is saved, and a new segment is started, using input
        point as end.
        """

        # Do not accept less or more dimensions
        if len(vRaw) != self.dimensions_count:
            raise TypeError(f"This FLInD instance only accepts {self.dimensions_count}-dimension tuples.")

        # First add init
        if self.firstStartingTimeRaw == -1:
            self.firstStartingTimeRaw = tRaw
            self._initModel(tRaw, vRaw)
            return

        # Check if all dimensions fit
        tNorm: float = tRaw - self.currentStartingTimeRaw
        dNorms = np.array(vRaw) - self.currentStartingValues
        dSlopes = dNorms / tNorm
        everybodyFits = np.greater_equal(self.maxPossibleSlopes, dSlopes).all() and np.greater_equal(dSlopes, self.minPossibleSlopes).all()

        # Update upper and lower points if all dimensions fit, and return
        if everybodyFits:
            self.lastTimeRaw = tRaw
            self.minPossibleSlopes = np.maximum(self.minPossibleSlopes, (dNorms - self.errors) / tNorm)
            self.maxPossibleSlopes = np.minimum(self.maxPossibleSlopes, (dNorms + self.errors) / tNorm)
            self.lastRawValues = np.array(vRaw)
            self.currentSlopes = np.array(dSlopes)
            return

        # Else, store current in-memory data point...
        if len(self.listOldModels) == 1 and self.listOldModels[0][0] == self.currentStartingTimeRaw:
            print('todo: fix first model being stored twice')
        else:
            self.listOldModels.append(np.append([self.currentStartingTimeRaw], self.currentStartingValues))

        # ...and use input point as new segment ending
        self.currentStartingTimeRaw = self.lastTimeRaw
        self.currentStartingValues = self.lastRawValues

        tNorm = tRaw - self.lastTimeRaw
        dNorms = np.array(vRaw) - self.lastRawValues
        dSlopes = dNorms / tNorm
        self.minPossibleSlopes = (dNorms - self.errors) / tNorm
        self.maxPossibleSlopes = (dNorms + self.errors) / tNorm
        self.currentSlopes = dSlopes
        self.lastTimeRaw = tRaw
        self.lastRawValues = np.array(vRaw)

    def add_frame(self, frame: pl.DataFrame, series: [str], sample_size: int = -1) -> None:
        """
        Convenience method loading the instance with a subpart of a dataframe.

        The subframe to be added to this model is computed by extracting the
        {series} columns from the {frame} DataFrame.
        If the entire frame should not be loaded in the instance, {sample_size}
        allows to only extract the first {sample_size} points.
        """

        limit = len(frame) if sample_size == -1 else sample_size
        subframe = frame.select(['Time'] + series).limit(limit)

        with mo.status.progress_bar(total=limit, subtitle='Loading model with data...', remove_on_exit=True) as bar:
            for row in subframe.iter_rows():
                time = row[0]
                value = row[1:]
                self.add(time, value)
                bar.update()

    def read(self, t: float) -> tuple:
        """
        Returns values associated to input {t}.

        This looks for a data point matching input time {t} and returns it if it
        exists; if not, this retrieves both previous and next points, and performs
        a linear regression for each dimension.

        If {t} is higher than the timestamp of the last stored point, a data point
        will be computed using upper and lower bounds for each dimension.
        """

        # Cannot return values when not initialized, or if asked about the past
        if self.firstStartingTimeRaw == -1 or t < self.firstStartingTimeRaw:
            raise IndexError('Error the time is unavailable')

        # Prediction using in-memory upper/lower bounds
        if self.currentStartingTimeRaw <= t:
            return self._getValue(t - self.currentStartingTimeRaw, self.currentSlopes, self.currentStartingValues)

        # If prediction is not possible, cannot interpolate with no stored points
        if len(self.listOldModels) == 0:
            raise IndexError('Error the time is unavailable (!)')

        # Retrieve models to be used in slopes computing
        _index: int = self._getIndexRead(t)
        model = self.listOldModels[_index]
        assert len(model) == self.segment_length

        # Prepare parameters for linear regression
        startingValues = model[1:]
        isLastModel = _index == len(self.listOldModels) - 1
        nextStartingValues = self.currentStartingValues if isLastModel else self.listOldModels[_index+1][1:]
        nextTime = self.currentStartingTimeRaw if isLastModel else self.listOldModels[_index+1][0]
        tNorm = nextTime - model[0]

        #debug
        if tNorm == 0:
            dprint('todo: fix first read time')
            tNorm = 0.001

        # Compute slopes and perform linear regression
        slopes = (nextStartingValues - startingValues) / tNorm
        return self._getValue(t - model[0], slopes, startingValues)

    def _getValue(self, t: float, slopes: [float], startingValues: [float]) -> tuple:
        """
        Internal method computing the data point associated to input {t}.

        This performs the linear regression `ax + b` operation for each data point
        dimension.
        """

        return ([startingValues[i] + t * slopes[i] for i in range(len(slopes))])

    def _getIndexRead(self, t: float) -> int:
        """
        Internal method returning the index of the point matching input {time}.
        """

        minIndex: int = 0
        maxIndex: int = len(self.listOldModels) - 1
        _index: int = -1
        tIndex: float
        while minIndex <= maxIndex:
            _index = (maxIndex + minIndex) // 2
            if minIndex == maxIndex:
                break
            tIndex = self.listOldModels[_index][0]
            if t == tIndex:
                break
            if t > tIndex and t < self.listOldModels[_index + 1][0]:
                break
            if t < tIndex:
                maxIndex = _index - 1
            else:
                minIndex = _index + 1
        return _index

    # Utils to benchmark models together

    def is_lossless(self) -> bool:
        return False

    def requires_batch_compression(self) -> bool:
        return False

    def batch_compress(self) -> None:
        pass


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Segment format

    * Timestamp
    * For each modelled dimension:
        * Starting value;

    {
    mo.md(f'For instance, with 2 dimensions, time series would be stored using the following format:<br/>$(timestamp, lat_{{start\\_value}}, lng_{{start\\_value}})$').callout(kind='neutral')
    }

    Slopes are needed to read values from linear interpolations, but these can be recomputed using both previous and next stored values.

    ## Choosing the series

    ### Why is it needed?

    Despite FLInD being designed to host multiple time series, one cannot blindly assign multiple series to it and expect it to have good performance.
    FLInD's performance is best when modelled series trigger point retention at the same time, leading to FLIND preserving a lesser amount of points.
    """)
    return


@app.cell(hide_code=True)
def _():
    offset_slider = mo.ui.slider(-30, 40, 1, value=2, debounce=True, label="Orange series offset:", show_value=True)
    corr_error_slider = mo.ui.slider(0, 5, 0.1, value=0.2, debounce=True, label="Model error:", show_value=True)

    def get_corr_main_series():
        t = [i for i in range(51)]
        y = [0 for i in range(10)]
        y += [i for i in range(5)]
        y += [i for i in range(5, -5, -1)]
        y += [i for i in range(-5, 1, 1)]
        y += [0 for i in range(20)]
        return (t, y)

    def get_corr_offset_series(t, y, offset):
        t1 = t.copy()
        y1 = y.copy()
        if offset >= 0:
            y1 = [0 for i in range(offset)] + y[:-offset or len(y)]
        else:
            offset = -1 * offset
            y1 = y[offset:] + [0 for i in range(offset)]
        return (t1, y1)
    return (
        corr_error_slider,
        get_corr_main_series,
        get_corr_offset_series,
        offset_slider,
    )


@app.cell(hide_code=True)
def _(
    corr_error_slider,
    get_corr_main_series,
    get_corr_offset_series,
    offset_slider,
):
    # Toy example
    def _():
        #reset_plt_font_scale()
        plt.rcParams.update({'font.size': 8})
        t, y = get_corr_main_series()
        t1, y1 = get_corr_offset_series(t, y, offset_slider.value)

        fig = plt.figure(figsize=[4, 2])
        plt.plot(t, y, marker='.', markersize=3)
        plt.plot(t1, y1, marker='.', markersize=3)

        # Model both series in FLInD
        model = FLInD(2)
        model.setError(corr_error_slider.value)
        for i in range(len(t)):
            model.add(t[i], (y[i], y1[i]))
        size = model.complete_size()

        # Display model breaks
        complete_data = model.complete_data()
        plt.vlines([s[0] for s in complete_data], -5, 5, 'red', linewidth=0.5, linestyles='--')

        # Assign axes labels
        ax = plt.gca()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        return mo.vstack([
                mo.md('To better understand this phenomenon, let\'s take a look at a toy example.<br/><br/>'),
                mo.md('On this figure, two series are displayed in blue and orange; the orange one can be offsetted using the input slider below.'),
                mo.md('FLInD allows retaining a limited quantity of data points using an error threshold: red lines mark timestamps for the points that are preserved.<br/>Whether said points\' timestamps are shared between series directly influence model size.<br/><br/>'),
                #corr_error_slider,
                mo.hstack([
                    ax,
                    mo.vstack([
                        offset_slider,
                        mo.md(f'<br/>Storage cost for the current model is **{int(size / 8)} floats**:<br/>- *Point size:* one timestamp float + 1 float for each series = **3**<br/>- *Points count:* **{len(complete_data)}**'),
                        mo.md('<br/><u>Model size is optimal when both series trigger point retention at the same time.</u>')
                    ], align='start', justify='start', gap=0)
                ], widths=[4.2, 7], align='start', justify='start', wrap=True)
            ])
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ### Methodology

    Since series timestamps are representative of how well FLInD will model them, we propose to use them **before** modelling series to determine whether series can be modelled together in FLInD.<br/>
    More specifically, when comparing two series, we:

    * *(i)* model each series with FLIND separately in a lossless manner
    * *(ii)* collect each series timestamps and
    * *(iii)* compute the ratio of all timestamps intersection over all timestamps union.

    The outputted quantity, in $[0; 1]$, describes how good FLIND will be in modelling both series together.<br/>
    Repeating this process
    over multiple series, we can build the shared timestamp ratios matrix, and exploring it with clustering algorithms, we can identify groups of well-suited series that will model well together.

    ---

    Let's use system metrics series {cite('kent_madsen_2022')} as an example here.

    We compute the shared timestamps ratios matrix over a sample of five series from the dataset:

    * CPU Core #1
    * CPU Core #2
    * CPU Core #3
    * CPU Core #4
    * GPU Memory

    Only the first 200 samples of each series are used to speed up the process.
    """)
    return


@app.function(hide_code=True)
# Compute matrix of timestamps ratios
def compute_timestamps_matrix(frame: pl.DataFrame, columns: [str], sample_size: int, model_error: Union[float, [float]], is_big_matrix: bool) -> [[float]]:
    series_breakpoints = []
    l = len(columns)
    is_errors_array = isinstance(model_error, list)
    dprint(f'is_errors_array: {is_errors_array}')

    with mo.status.progress_bar(total=l, title='Collecting preserved timestamps') as bar:
        for index, col in enumerate(columns):
            bar.update(subtitle=f'Computing "{col}" series...')
            model = FLInD(1)
            error = model_error if not is_errors_array else model_error[index]
            model.setError(error)
            model.add_frame(frame, [col], sample_size)
            breakpoints = [p[0] for p in model.complete_data()]
            series_breakpoints.append(np.array(breakpoints))

    matrix = []
    with mo.status.progress_bar(total=(l*l/2)-(l/2), title='Looking for shared timestamps') as bar:
        for c_index, col in enumerate(columns):
            line = []

            for cc_index, ccol in enumerate(columns):
                if c_index <= cc_index:
                    line.append(1 if c_index == cc_index else 0)
                    continue

                bar.update(subtitle=f'Comparing "{col}" and "{ccol}" series...')
                intersection = np.intersect1d(series_breakpoints[c_index], series_breakpoints[cc_index])
                union = np.union1d(series_breakpoints[c_index], series_breakpoints[cc_index])
                ratio = len(intersection) / len(union)
                line.append(round(ratio, 1) if is_big_matrix == True else ratio)

            matrix.append(line)

    return matrix


@app.cell(hide_code=True)
def _(hardware_profile_df):
    toy_system_columns = ['CPU Core #1', 'CPU Core #2', 'CPU Core #3', 'CPU Core #4', 'GPU Memory']
    def _():
        sample_size = 200
        model_error = 0
        cols = list(sorted(toy_system_columns))
        is_big_matrix = len(cols) > 10

        matrix = compute_timestamps_matrix(hardware_profile_df, cols, sample_size, model_error, is_big_matrix)
        plot = display_matrix(matrix, cols, False, is_big_matrix)
        return plot

    _()
    return (toy_system_columns,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It seems like all CPU series seem to share most of their timestamps.<br/>
    On the other hand, the GPU memory series seems incompatible with CPU series.

    Converting the matrix to a graph, we can easily extract groups of series to be used later in modelling.
    For instance, with the same subset of series, we compute the following matrix:
    """)
    return


@app.function(hide_code=True)
# Display matrix logic
def display_matrix(matrix, cols, hide_labels: bool = False, is_big_matrix: bool = False, color_scale = sns.color_palette("YlOrBr", as_cmap=True), export: str = ''):
    """
    Displays a matrix as a heatmap.

    {cols} is the list of columns of the dataset, {hide_labels} determines whether
    to display text labels over the heatmap (with a big enough count of columns
    text overlaps, so you might want to hide it).
    """

    if is_big_matrix:
        plt.rcParams.update({'font.size': 0.1})
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=2000)
    else:
        plt.rcParams.update({'font.size': 5})
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=200)

    sns.heatmap(matrix, cmap=color_scale, cbar=False, annot=True, xticklabels=cols, yticklabels=cols, ax=ax)

    # Hide texts for big matrixes
    if hide_labels:
        for text in ax.texts:
            text.set_visible(False)

    if len(export) > 0:
        plt.savefig(f'{export}.pdf', bbox_inches='tight')
        dprint(f'Figure exported to {export}.pdf.')
        return plt.gca()
    else:
        return plt.gca()


@app.function(hide_code=True)
# Extract clusters through graph
def clusters_search(matrix: [[float]], series: [str], threshold: float) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(series)
    for index, row in enumerate(matrix):
        for l_index, value in enumerate(row):
            # don't check matrix top right
            if l_index >= index:
                continue
            if value < threshold:
                continue
            graph.add_edge(series[index], series[l_index])
    return graph


@app.function(hide_code=True)
def display_graph(graph, highlighted_series: [str] = []):
    # Available color scales: https://matplotlib.org/stable/users/explain/colors/colormaps.html#cyclic
    color_scale = plt.cm.PuBuGn

    node_options = {
        "node_size": 20,
        "node_color": range(graph.number_of_nodes()),
        "cmap": color_scale,
        "edgecolors": "black",
        "linewidths": 0.3,
    }

    edge_options = {
        "width": 0.2
    }

    label_options = {
        "font_size": 5
    }

    fig = plt.figure(figsize=(2, 2), dpi=400)
    ax = plt.gca()
    pos = nx.spring_layout(graph, seed=42, store_pos_as="pos")
    #pos = nx.forceatlas2_layout(graph, seed=42, store_pos_as="pos")
    #pos = nx.kamada_kawai_layout(graph, store_pos_as="pos")
    #pos = nx.spiral_layout(graph, store_pos_as="pos")

    # Cool layout used in paper, but requires dependencies to be installed on the host machine
    ## https://pygraphviz.github.io/documentation/stable/install.html
    #pos = nx.nx_agraph.graphviz_layout(graph)

    # Optionally highlight series
    if len(highlighted_series) > 0:
        # Compute non-clique series
        series = list(graph.nodes())
        for s in highlighted_series:
            series.remove(s)

        # Split edges
        clique_edges = []
        edges = []
        for edge in graph.edges():
            if edge[0] in highlighted_series and edge[1] in highlighted_series:
                clique_edges.append(edge)
            else:
                edges.append(edge)

        # Highlight style
        highlighted_node_options = {
            "node_size": 20,
            "node_color": "green",
            "edgecolors": "darkgreen",
            "linewidths": 0.6,
        }
        highlighted_edge_options = {
            "width": 0.4,
            "edge_color": "green"
        }

        # Unhighlighted style
        node_options = {
            "node_size": 20,
            "node_color": 'white',
            "edgecolors": "black",
            "linewidths": 0.3,
        }

        nx.draw_networkx_nodes(graph, pos, nodelist=series, **node_options)
        nx.draw_networkx_edges(graph, pos, edgelist=edges, **edge_options)
        nx.draw_networkx_nodes(graph, pos, nodelist=highlighted_series, **highlighted_node_options)
        nx.draw_networkx_edges(graph, pos, edgelist=clique_edges, **highlighted_edge_options)
    else:
        nx.draw_networkx_nodes(graph, pos, **node_options)
        nx.draw_networkx_edges(graph, pos, **edge_options)

    # Slightly offset labels
    # pos_higher = {}
    # y_off = 15
    # for k, v in pos.items():
    #     pos_higher[k] = (v[0], v[1]+y_off)
    # labels = nx.draw_networkx_labels(graph, pos_higher, verticalalignment='top', **label_options)

    ax.margins(0.20)
    # plt.axis("off")

    return ax


@app.function(hide_code=True)
# Visualise all steps of the clusters search process
def get_clustering_steps(frame: pl.DataFrame, columns: [str], sample_size, model_error, view_carousel: bool = True, widths = [2, 1.8], threshold = 0.1):
    is_big_matrix = len(columns) > 10
    columns = list(sorted(columns))

    # Compute initial shared timestamps ratios matrix
    matrix = compute_timestamps_matrix(frame, columns, sample_size, model_error, is_big_matrix)

    # Compute visual output
    with mo.status.spinner(title='Compute matrix render', subtitle="Building heatmap ...") as spinner:
        spinner.update()
        plot = None

        # Don't compute matrix for high dimensional series
        if len(matrix) > 110:
            plt.title('Not displaying big matrix')
            plot = plt.gca()
        else:
            plot = display_matrix(matrix, columns, is_big_matrix, is_big_matrix)

        # Convert to graph excluding values below a threshold
        spinner.update(title='Convert matrix to graph', subtitle="Building graph ...")
        graph = clusters_search(matrix, columns, threshold)

        spinner.update(subtitle="Exploring for cliques...")
        # series_groups = list(nx.connected_components(graph))
        # series_groups = list(nx.algorithms.clique.find_cliques_recursive(graph))
        series_groups = list(nx.algorithms.clique.find_cliques(graph))
        best_group = list(sorted(max(series_groups, key=len)))
        series_groups_len = len(series_groups)
        series_groups_str = 'no clique' if series_groups_len == 0 else 'one clique' if series_groups_len == 1 else f'{series_groups_len} cliques'

        return series_groups, (mo.carousel([
            mo.hstack([
                mo.vstack([
                    mo.md('### Initial breakpoint ratio matrix'),
                    mo.md('This matrix displays the ratio of all timestamps intersection over all timestamps union for all input series pairs.')
                ]),
                plot
            ], widths=widths),
            mo.hstack([
                mo.vstack([
                    mo.md('### Conversion to graph'),
                    mo.md(f'Then, we convert the shared breakpoints matrix into a graph, representing each series with a node, and creating edges between nodes only if their shared breakpoints rate is above a given threshold.'),
                    mo.md(f'Using **{threshold}** as threshold value, *{series_groups_str}* can be extracted from the graph.')
                ]),
                display_graph(graph)
            ], widths=widths),
            mo.hstack([
                mo.vstack([
                    mo.md('### Finding the largest clique'),
                    mo.md('We then look for the biggest clique, *e.g.* the one that contains the most series to model, and color it in green.'),
                    mo.md(f'Here, the biggest clique encompasses the following {len(best_group)} series:<br/>')
                ] + [
                    mo.md("\n".join([f'- {series}' for series in best_group])),
                ]),
                display_graph(graph, best_group)
            ], widths=widths),
        ]) if view_carousel else None)


@app.cell(hide_code=True)
def _(hardware_profile_df, toy_system_columns):
    def _():
        sample_size = 200
        model_error = 0
        return get_clustering_steps(hardware_profile_df, toy_system_columns, sample_size, model_error)
    toy_cliques, toy_carousel = _()
    toy_carousel
    return (toy_cliques,)


@app.cell(hide_code=True)
def _(hardware_profile_df, toy_cliques):
    def _():
        labels = list(max(toy_cliques, key=len))
        dimensions_count = len(labels)
        frames = bench_dimensional_models([FLInD(dimensions_count), FLInDwFLI(dimensions_count), RawStorage(dimensions_count)], hardware_profile_df, labels, [0, 0, 0])

        return mo.vstack([
            mo.md(f"Before taking advantage of cliques, let us model the series of the biggest clique."),
            format_dimensional_bench_results(frames, labels),
            mo.md('Some interesting facts here:'),
            mo.md('* Modelling the series separately is worse than storing data as-is;'),
            mo.md('* When configured with a tolerated error of 0 (worse case), FLInD is equivalent to raw storage.'),
        ]).callout(kind='info')
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now compare different ways of modelling data:

    1. Storing all series separately;
    2. Storing all series in a single FLInD instance;
    3. Storing top clique series in a FLInD instance, and the other separately.
    """)
    return


@app.cell(hide_code=True)
def _(hardware_profile_df, toy_cliques, toy_system_columns):
    def _():
        # Model all series individually, and with FLInD
        l = len(toy_system_columns)
        frames = bench_dimensional_models([FLInD(l), FLInDwFLI(l)], hardware_profile_df, toy_system_columns, [0, 0])
        result = frames[0]
        all_flind_size = result['Size'][0]
        all_fli_size = result['Size'][1]


        # Model the cluster with FLInD and the rest individually
        # (should be better than first stats)

        ## FLInD-modelling of the biggest found cluster
        labels = list(max(toy_cliques, key=len))
        l = len(labels)
        frames = bench_dimensional_models([FLInD(l)], hardware_profile_df, labels, [0])
        flind_part_size = frames[0]['Size'][0]

        ## Separated modelling of other series
        other_series = l3 = [series for series in toy_system_columns if series not in labels]
        l = len(other_series)
        frames = bench_dimensional_models([FLInDwFLI(l)], hardware_profile_df, other_series, [0])
        separated_part_size = frames[0]['Size'][0]


        # Layout
        return mo.md(f"""
        Here are the size results of modelling all series of the hardware profile dataset:

        * Modelling each series separately: **{human_readable_size(all_fli_size)}**;
        * Modelling all series with FLInD: **{human_readable_size(all_flind_size)}**;
        * Modelling biggest clique series with FLInD, and other series separately: **{human_readable_size(flind_part_size + separated_part_size)}**.
        """)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # Experimental setup

    To ease comparison with other models, we implement the clique search process into a new model, `CliqueFLInD`.
    """)
    return


@app.class_definition(hide_code=True)
class CliqueFLInD(FLInD):
    def __init__(self, dimensions_count, clique_sample_size = 200, threshold = 0.1, errors = []):
        self.dimensions_count = dimensions_count
        self.clique_sample_size = clique_sample_size
        self.threshold = threshold
        self.combined_model = None
        self.separated_model = None
        self.combined_series_indexes = []
        FLInD.__init__(self, dimensions_count, errors)

    def _models(self):
        return [self.combined_model, self.separated_model]

    def setError(self, value: float):
        self._tmp_err = value

    def name(self):
        return f'CliqueFLInD (n={self.dimensions_count})'

    def data(self):
        d = []
        for m in self._models():
            d += m.data()
        return d

    def size(self):
        total = 0
        for m in self._models():
            total += m.size()
        return total

    def add_frame(self, frame: pl.DataFrame, series: [str], sample_size: int = -1) -> None:
        limit = len(frame) if sample_size == -1 else sample_size
        subframe = frame.select(['Time'] + series).limit(limit)

        # Compute series repartition
        dprint(f'Looking for cliques (sample_size: {self.clique_sample_size}, threshold: {self.threshold})')
        matrix = compute_timestamps_matrix(frame, series, self.clique_sample_size, self.maxError, True)
        graph = clusters_search(matrix, series, self.threshold)
        series_groups = list(nx.algorithms.clique.find_cliques(graph))
        best_group = list(sorted(max(series_groups, key=len)))
        self.combined_series_indexes = [series.index(s) for s in best_group]

        # errors splitting
        combined_errors = []
        separated_errors = []
        for index, value in enumerate(self.errors):
            if index in self.combined_series_indexes:
                combined_errors.append(index)
            else:
                separated_errors.append(index)

        l = len(self.combined_series_indexes)
        ll = len(series) - l
        self.combined_model = FLInD(l, combined_errors)
        self.separated_model = FLInDwFLI(ll, separated_errors)
        self.combined_model.setError(combined_errors)
        dprint(f'Storing {l} series together, and {ll} separately.')

        with mo.status.progress_bar(total=limit, subtitle='Loading model with data...', remove_on_exit=True) as bar:
            for row in subframe.iter_rows():
                time = row[0]
                value = row[1:]
                self.add(time, value)
                bar.update()

    def add(self, tRaw: float, vRaw: tuple):
        # do not accept less or more dimensions
        if len(vRaw) != self.dimensions_count:
            raise TypeError(f"This FLInDwFLI instance only accepts {self.dimensions_count}-dimension tuples.")

        # do not accept data before computing series repartition
        if len(self.combined_series_indexes) == 0:
            raise Exception('Cannot store data before computing series repartition.')

        # split data point between two models
        #todo: optimize?
        combined_point = tuple([v for i, v in list(enumerate(vRaw)) if i in self.combined_series_indexes])
        self.combined_model.add(tRaw, combined_point)
        separated_point = tuple([v for i, v in list(enumerate(vRaw)) if i not in self.combined_series_indexes])
        self.separated_model.add(tRaw, separated_point)

    def read(self, t: float) -> tuple:
        combined_series_point = list(reversed(self.combined_model.read(t)))
        separated_series_point = list(reversed(self.separated_model.read(t)))
        new_point = []
        for index in range(self.dimensions_count):
            if index in self.combined_series_indexes:
                new_point.append(combined_series_point.pop())
            else:
                new_point.append(separated_series_point.pop())
        return tuple(new_point)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Competitors

    #### Fast Linear Interpolation

    * n dimension implemen w one FLI per dim
    """)
    return


@app.class_definition(hide_code=True)
class FLInDwFLI(FLInD):
    def __init__(self, dimensions_count, errors = []):
        self.dimensions_count = dimensions_count
        self.fli_models = [FastLinearInterpolation() for i in range(dimensions_count)]
        FLInD.__init__(self, dimensions_count, errors)

    def setError(self, value: float):
        # Classic case
        if len(self.errors) == 0:
            for m in self.fli_models:
                m.setError(value)

        # Using input array
        else:
            for i in range(self.dimensions_count):
                self.fli_models[i].setError(self.errors[i])

    def name(self):
        return f'FLI (FLInDwFLI[n={self.dimensions_count}])'

    def data(self):
        d = []
        for m in self.fli_models:
            d += m.data()
        return d

    def size(self):
        total = 0
        for m in self.fli_models:
            total += m.size()
        return total

    def add(self, tRaw: float, vRaw: tuple):
        # do not accept less or more dimensions
        if len(vRaw) != self.dimensions_count:
            raise TypeError(f"This FLInDwFLI instance only accepts {self.dimensions_count}-dimension tuples.")

        for index, element in enumerate(vRaw):
            self.fli_models[index].add(tRaw, element)

    def read(self, t: float) -> tuple:
        return [m.read(t) for m in self.fli_models]

    def points_per_segment(self):
        return self.latModel.points_per_segment() + self.lngModel.points_per_segment()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Raw storage
    """)
    return


@app.class_definition(hide_code=True)
class RawStorage(FLInD):
    def __init__(self, dimensions_count):
        self.dimensions_count = dimensions_count
        self.dict = {}
        FLInD.__init__(self, dimensions_count)

    def setError(self, value: float):
        pass

    def name(self):
        return f'Raw storage'

    def data(self):
        return self.dict

    def size(self):
        return (1+self.dimensions_count) * len(self.dict) * 8

    def add(self, tRaw: float, vRaw: tuple):
        # do not accept less or more dimensions
        if len(vRaw) != self.dimensions_count:
            raise TypeError(f"This {self.name()} instance only accepts {self.dimensions_count}-dimension tuples.")
        self.dict[tRaw] = vRaw

    def read(self, t: float) -> tuple:
        return self.dict[t]

    def points_per_segment(self):
        return -1

    def is_lossless(self):
        return True


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Zip
    """)
    return


@app.class_definition(hide_code=True)
class ZipStorage(FLInD):
    def __init__(self, dimensions_count):
        self.dimensions_count = dimensions_count
        self.frame = pl.DataFrame()
        FLInD.__init__(self, dimensions_count)

    def _models(self):
        return []

    def setError(self, value: float):
        pass

    def name(self):
        return 'ZipStorage'

    def data(self):
        return []

    def size(self):
        zp = zipfile.ZipFile(self.zip_filename)
        total_size = sum([zinfo.file_size for zinfo in zp.filelist])
        return total_size

    def add_frame(self, frame: pl.DataFrame, series: [str], sample_size: int = -1) -> None:
        # Export series to a CSV file
        filename = f'{tempfile.NamedTemporaryFile(delete=False).name}.csv'
        limit = len(frame) if sample_size == -1 else sample_size
        subframe = frame.select(['Time'] + series).limit(limit)
        subframe.write_csv(filename)

        # Compress CSV file
        zipfilename = f'{tempfile.NamedTemporaryFile(delete=False).name}.zip'
        zf = zipfile.ZipFile(zipfilename, mode='w')
        try:
            zf.write(filename, compress_type=zipfile.ZIP_DEFLATED)
        finally:
            zf.close()

        self.csv_filename = Path(filename).name
        self.zip_filename = zipfilename

    def add(self, tRaw: float, vRaw: tuple):
        raise "Unimplemented"

    def read(self, t: float) -> tuple:
        """
        While the disk storage space of zip compression is lower than the
        in-memory budget of FLInD, how should we count the memory storage
        cost of zip? How to compute read/write performances ?

            1) For each read, look for the CSV file in the zip archive, open
            it without decompressing archive, and process it iteratively
            until correct timestamp is found;

            2) For each read, look for the CSV file in the zip archive, open
            it without decompressing archive, load it into a DataFrame kept
            in memory, then use this dataframe for subsequent reads.

        While the first possibility is very slow, it presents the advantage
        of not requiring a DataFrame, which in-memory cost should be then
        taken into account when computing the storage cost of `ZipStorage`.
        """

        # 1) open the archive on each read
        # zf = zipfile.ZipFile(self.zip_filename, 'r')
        # with zf.open(f'tmp/{self.csv_filename}', 'r') as infile:
        #     reader = csv.reader(TextIOWrapper(infile, 'utf-8'))
        #     for row in reader:
        #         if row[0] == 'Time':
        #             continue

        #         time = float(row[0])
        #         if time == t:
        #             return tuple([float(d) for d in row[1:]])

        # 2) open the archive on first read, and store its content in memory
        if self.frame.is_empty():
            zf = zipfile.ZipFile(self.zip_filename, 'r')
            self.frame = pl.read_csv(zf.open(f'tmp/{self.csv_filename}'))
        return self.frame.filter(pl.col('Time') == t).select(pl.exclude('Time')).row(0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### TimescaleDB
    """)
    return


@app.class_definition(hide_code=True)
class TimescaleDB(FLInD):
    def __init__(self, series):
        self.series = series
        self.dimensions_count = len(series)
        FLInD.__init__(self, self.dimensions_count)

        # Init database connection
        info = "postgres://postgres:password@127.0.0.1:5432/postgres"
        self.connection = psycopg2.connect(info)
        self.cursor = self.connection.cursor()

        # Eventually remove previous table
        table_removal_query = 'DROP TABLE IF EXISTS data CASCADE;'
        self.cursor.execute(table_removal_query)
        self.connection.commit()

        # Create table to host dataset
        table_creation_query = 'CREATE TABLE data (\n    time bigint'#TIMESTAMPTZ NOT NULL'
        for series in self.series:
            table_creation_query += f',\n    "{series}" DOUBLE PRECISION'
        table_creation_query += '\n);'
        self.cursor.execute(table_creation_query)
        self.connection.commit()

        # Build insert query for later write operations
        query = f'INSERT INTO data VALUES ({', '.join(['%s' for i in range(self.dimensions_count + 1)])});'
        self.insert_query_template = query

    def setError(self, value: float):
        pass

    def name(self):
        return f'TimescaleDB'

    def data(self):
        return {}

    def size(self):
        """
        `pg_total_relation_size` takes also into account database indexes, as
        opposed to `pg_relation_size`.
        """

        size_query = "SELECT pg_total_relation_size('data');"
        self.cursor.execute(size_query)
        self.connection.commit()
        result = self.cursor.fetchone()
        return result[0]

    def add(self, tRaw: float, vRaw: tuple):
        # Do not accept less or more dimensions
        if len(vRaw) != self.dimensions_count:
            raise TypeError(f"This {self.name()} instance only accepts {self.dimensions_count}-dimension tuples.")

        # SQL insert
        try:
            self.cursor.execute(self.insert_query_template, (tRaw,) + vRaw)
        except (Exception) as error:
            dprint(f'Exception caught: {error}')

    def read(self, t: float) -> tuple:
        read_query = f'SELECT * FROM data where time = {t};'
        self.cursor.execute(read_query)
        result = self.cursor.fetchone()
        self.connection.commit()
        return result[1:] # Excluding time field

    def points_per_segment(self):
        return -1

    def is_lossless(self):
        return True


@app.cell(hide_code=True)
def _(hardware_profile_columns, hardware_profile_df):
    # Implementation testing
    def _():
        return
        model = TimescaleDB(hardware_profile_columns)
        model.add_frame(hardware_profile_df, hardware_profile_columns)
        print(model.size())
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### TDengine
    """)
    return


@app.cell(hide_code=True)
def _():
    import datetime
    import os
    from taosrest import TaosRestCursor

    class TDengine(FLInD):
        def __init__(self, series, data_folder = "~/tmp/tdengine/data/"):
            self.series = series
            self.data_folder = data_folder
            self.dimensions_count = len(series)
            FLInD.__init__(self, self.dimensions_count)

            # Init database connection
            self.connection = taosrest.connect(url="127.0.0.1:6041", token="")
            self.cursor: TaosRestCursor = self.connection.cursor()

            # Create database
            self.cursor.execute("CREATE DATABASE IF NOT EXISTS tdengine;")

            # Eventually remove previous table
            self.cursor.execute("USE tdengine;")
            table_removal_query = 'DROP TABLE IF EXISTS tdengine.data;'
            self.cursor.execute(table_removal_query)

            # Create table to host dataset
            table_creation_query = 'CREATE TABLE tdengine.data (\n    time timestamp'
            for series in self.series:
                table_creation_query += f',\n     {self.getTDengineSeriesName(series)} float'
            table_creation_query += '\n);'
            self.cursor.execute(table_creation_query)

        def getTDengineSeriesName(self, series) -> str:
            """
            TDengine features strict rules as naming conventions, meaning series names
            cannot be used as column names directly.
            More info:
            https://docs.tdengine.com/tdengine-reference/sql-manual/names/
            """

            return f'_{series.replace(' ','_').replace('#','')}'

        def getTDengineTimestamp(self, epoch) -> str:
            """
            Same here.
            https://docs.tdengine.com/tdengine-reference/sql-manual/data-types/#timestamp
            """
            t = datetime.datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            return f'"{t}"'

        def setError(self, value: float):
            pass

        def name(self):
            return f'TDengine'

        def data(self):
            return {}

        def size(self):
            #https://stackoverflow.com/questions/69966205/how-can-i-check-the-size-of-a-particular-database-in-tdengine

            # Retrive vnode ids
            size_query = 'show tdengine.vgroups;'
            self.cursor.execute(size_query)
            result = self.cursor.fetchall()
            node_ids = [r[0] for r in result]

            # Get disk size for each vnode
            total_size = 0
            for node_id in node_ids:
                path = f'{self.data_folder}vnode/vnode{node_id}'
                if '~' in path:
                    path = os.path.expanduser(path)
                path = Path(path)
                total_size += sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

            return total_size

        def add(self, tRaw: float, vRaw: tuple):
            # Do not accept less or more dimensions
            if len(vRaw) != self.dimensions_count:
                raise TypeError(f"This {self.name()} instance only accepts {self.dimensions_count}-dimension tuples.")

            # SQL insert
            try:
                x = ', '.join([str(x) for x in list((self.getTDengineTimestamp(tRaw),) + vRaw)])
                insert_query = f'INSERT INTO tdengine.data VALUES ({x});'
                self.cursor.execute(insert_query)
            except (Exception) as error:
                dprint(f'Exception caught: {error}')

        def read(self, t: float) -> tuple:
            read_query = f'SELECT * FROM tdengine.data where time = {self.getTDengineTimestamp(t)};'
            self.cursor.execute(read_query)
            result = self.cursor.fetchone()
            #self.connection.commit()
            return result[1:] # Excluding time field

        def points_per_segment(self):
            return -1

        def is_lossless(self):
            return True
    return (TDengine,)


@app.cell(hide_code=True)
def _(TDengine, hardware_profile_columns, hardware_profile_df):
    # Implementation testing
    def _():
        t = hardware_profile_df.rows()[0][0]
        #return
        model = TDengine(hardware_profile_columns)
        model.add_frame(hardware_profile_df, hardware_profile_columns, 1000)
        print(model.size())
        print(model.read(t))
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Datasets

    ### Hardware performance profile

    This dataset {cite('kent_madsen_2022')} includes system metrics, like CPU or memory usage, of a laptop over several days of of normal functioning. In this study, we select data from the biggest file (2019/05/04) including 77,070 records, sampled every second.
    """)
    return


@app.cell(hide_code=True)
def _():
    hardware_profile_df = pl.read_csv("data/OpenHardwareMonitorLog-2019-05-04.csv", skip_rows=1, infer_schema_length=100000).with_columns(pl.col("Time").str.to_datetime().dt.timestamp("ms") / 1000).sort(by='Time').unique(subset='Time') # import data frame converting datetime to timestamps
    hardware_profile_df
    return (hardware_profile_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Electricity consumption

    This dataset contains electricity consumption data of 370 clients in Portugal ([full details here](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)).

    Consumption is exprimed in kW, and is reported at a rate of one measurement every 15 minutes, covering from 2011 to 2014 (included).

    > Some clients were created after 2011. In these cases consumption were considered zero.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _():
        def ensure_file_format():
            """
            Ensures dataset file is ready to be used in experiments.

            The original electricity dataset file uses commas to represent floats,
            which is not supported by polars nor pandas; thus, this replaces
            commas with dots, and exports results to a new file.

            The comma replacing processus is skipped if the formatted file already
            exists.
            """

            file = Path('data/LD2011_2014.formatted.txt')
            if file.is_file():
                # No need to do anything
                return
            else:
                dprint('Downloading archive...')
                urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip", "data/electricityloaddiagrams20112014.zip")
                dprint('Unzipping archive...')
                with zipfile.ZipFile('data/electricityloaddiagrams20112014.zip', 'r') as zip_ref:
                    zip_ref.extractall('data')

            with open('data/LD2011_2014.txt') as f:
                s = f.read()
                s = s.replace(',', '.')
                with open('data/LD2011_2014.formatted.txt', mode='w') as ff:
                    ff.write(s)

        ensure_file_format()
        types = {}
        for i in range(1, 371):
            types["MT_" + str(i).zfill(3)] = pl.Float64
        return pl.read_csv('data/LD2011_2014.formatted.txt', separator=";", schema_overrides=types).with_columns(pl.col("").str.to_datetime().dt.timestamp("ms")).sort(by='').unique(subset='').rename({'': 'Time'})

    electricity_df = _()
    electricity_df
    return (electricity_df,)


@app.cell(hide_code=True)
def _():
    electricity_series = ['MT_001', 'MT_002', 'MT_003', 'MT_004', 'MT_005', 'MT_006', 'MT_007', 'MT_008', 'MT_009', 'MT_010', 'MT_011', 'MT_012', 'MT_013', 'MT_014', 'MT_015', 'MT_016', 'MT_017', 'MT_018', 'MT_019', 'MT_020', 'MT_021', 'MT_022', 'MT_023', 'MT_024', 'MT_025', 'MT_026', 'MT_027', 'MT_028', 'MT_029', 'MT_030', 'MT_031', 'MT_032', 'MT_033', 'MT_034', 'MT_035', 'MT_036', 'MT_037', 'MT_038', 'MT_039', 'MT_040', 'MT_041', 'MT_042', 'MT_043', 'MT_044', 'MT_045', 'MT_046', 'MT_047', 'MT_048', 'MT_049', 'MT_050', 'MT_051', 'MT_052', 'MT_053', 'MT_054', 'MT_055', 'MT_056', 'MT_057', 'MT_058', 'MT_059', 'MT_060', 'MT_061', 'MT_062', 'MT_063', 'MT_064', 'MT_065', 'MT_066', 'MT_067', 'MT_068', 'MT_069', 'MT_070', 'MT_071', 'MT_072', 'MT_073', 'MT_074', 'MT_075', 'MT_076', 'MT_077', 'MT_078', 'MT_079', 'MT_080', 'MT_081', 'MT_082', 'MT_083', 'MT_084', 'MT_085', 'MT_086', 'MT_087', 'MT_088', 'MT_089', 'MT_090', 'MT_091', 'MT_092', 'MT_093', 'MT_094', 'MT_095', 'MT_096', 'MT_097', 'MT_098', 'MT_099', 'MT_100', 'MT_101', 'MT_102', 'MT_103', 'MT_104', 'MT_105', 'MT_106', 'MT_107', 'MT_108', 'MT_109', 'MT_110', 'MT_111', 'MT_112', 'MT_113', 'MT_114', 'MT_115', 'MT_116', 'MT_117', 'MT_118', 'MT_119', 'MT_120', 'MT_121', 'MT_122', 'MT_123', 'MT_124', 'MT_125', 'MT_126', 'MT_127', 'MT_128', 'MT_129', 'MT_130', 'MT_131', 'MT_132', 'MT_133', 'MT_134', 'MT_135', 'MT_136', 'MT_137', 'MT_138', 'MT_139', 'MT_140', 'MT_141', 'MT_142', 'MT_143', 'MT_144', 'MT_145', 'MT_146', 'MT_147', 'MT_148', 'MT_149', 'MT_150', 'MT_151', 'MT_152', 'MT_153', 'MT_154', 'MT_155', 'MT_156', 'MT_157', 'MT_158', 'MT_159', 'MT_160', 'MT_161', 'MT_162', 'MT_163', 'MT_164', 'MT_165', 'MT_166', 'MT_167', 'MT_168', 'MT_169', 'MT_170', 'MT_171', 'MT_172', 'MT_173', 'MT_174', 'MT_175', 'MT_176', 'MT_177', 'MT_178', 'MT_179', 'MT_180', 'MT_181', 'MT_182', 'MT_183', 'MT_184', 'MT_185', 'MT_186', 'MT_187', 'MT_188', 'MT_189', 'MT_190', 'MT_191', 'MT_192', 'MT_193', 'MT_194', 'MT_195', 'MT_196', 'MT_197', 'MT_198', 'MT_199', 'MT_200', 'MT_201', 'MT_202', 'MT_203', 'MT_204', 'MT_205', 'MT_206', 'MT_207', 'MT_208', 'MT_209', 'MT_210', 'MT_211', 'MT_212', 'MT_213', 'MT_214', 'MT_215', 'MT_216', 'MT_217', 'MT_218', 'MT_219', 'MT_220', 'MT_221', 'MT_222', 'MT_223', 'MT_224', 'MT_225', 'MT_226', 'MT_227', 'MT_228', 'MT_229', 'MT_230', 'MT_231', 'MT_232', 'MT_233', 'MT_234', 'MT_235', 'MT_236', 'MT_237', 'MT_238', 'MT_239', 'MT_240', 'MT_241', 'MT_242', 'MT_243', 'MT_244', 'MT_245', 'MT_246', 'MT_247', 'MT_248', 'MT_249', 'MT_250', 'MT_251', 'MT_252', 'MT_253', 'MT_254', 'MT_255', 'MT_256', 'MT_257', 'MT_258', 'MT_259', 'MT_260', 'MT_261', 'MT_262', 'MT_263', 'MT_264', 'MT_265', 'MT_266', 'MT_267', 'MT_268', 'MT_269', 'MT_270', 'MT_271', 'MT_272', 'MT_273', 'MT_274', 'MT_275', 'MT_276', 'MT_277', 'MT_278', 'MT_279', 'MT_280', 'MT_281', 'MT_282', 'MT_283', 'MT_284', 'MT_285', 'MT_286', 'MT_287', 'MT_288', 'MT_289', 'MT_290', 'MT_291', 'MT_292', 'MT_293', 'MT_294', 'MT_295', 'MT_296', 'MT_297', 'MT_298', 'MT_299', 'MT_300', 'MT_301', 'MT_302', 'MT_303', 'MT_304', 'MT_305', 'MT_306', 'MT_307', 'MT_308', 'MT_309', 'MT_310', 'MT_311', 'MT_312', 'MT_313', 'MT_314', 'MT_315', 'MT_316', 'MT_317', 'MT_318', 'MT_319', 'MT_320', 'MT_321', 'MT_322', 'MT_323', 'MT_324', 'MT_325', 'MT_326', 'MT_327', 'MT_328', 'MT_329', 'MT_330', 'MT_331', 'MT_332', 'MT_333', 'MT_334', 'MT_335', 'MT_336', 'MT_337', 'MT_338', 'MT_339', 'MT_340', 'MT_341', 'MT_342', 'MT_343', 'MT_344', 'MT_345', 'MT_346', 'MT_347', 'MT_348', 'MT_349', 'MT_350', 'MT_351', 'MT_352', 'MT_353', 'MT_354', 'MT_355', 'MT_356', 'MT_357', 'MT_358', 'MT_359', 'MT_360', 'MT_361', 'MT_362', 'MT_363', 'MT_364', 'MT_365', 'MT_366', 'MT_367', 'MT_368', 'MT_369', 'MT_370']
    return (electricity_series,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Solar power data

    This dataset contains solar power production data for several stations across the USA ([full details here](https://www.nrel.gov/grid/solar-power-data)); in this study, we use data from the state of California.

    <!--, which you can [download here](https://www.nrel.gov/media/docs/libraries/grid/zip/ca-pv-2006.zip) (put the ZIP archive into the `/data` directory).-->

    Power produced by the 405 California solar stations is exprimed in MW, and is reported at a rate of one measurement every 5 minutes, covering the entire year of 2006.

    Since the dataset originally contains both actual and forecast (simulated) data, we only use actual data.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _():
        def fetch_pv_dataset_archive():
            file = Path('data/ca-pv-2006.zip')
            if file.is_file():
                dprint('PV archive has already been downloaded, skipping fetching.')
                return
            dprint('Downloading archive...')
            url = 'https://www.nrel.gov/docs/libraries/grid/ca-pv-2006.zip'
            r = requests.get(url)
            with open('data/ca-pv-2006.zip', 'wb') as outfile:
                outfile.write(r.content)
            dprint('Done downloading archive.')

        def extract_solar_files():
            file = Path('data/solar_power')
            if file.is_dir():
                # No need to do anything
                dprint('Archive has already been unzipped, skipping decompression.')
                return
            else:
                dprint('Unzipping archive...')
                with zipfile.ZipFile('data/ca-pv-2006.zip', 'r') as zip_ref:
                    zip_ref.extractall('data/solar_power')
                    dprint('Done unzipping.')

        def read_solar_data() -> pl.DataFrame:
            actual_files = glob.glob('data/solar_power/Actual*.csv')
            dprint(f'Building DataFrame from {len(actual_files)} CSV files...')
            master_frame = pl.DataFrame()

            for file in actual_files:
                words = file.split('_')
                latitude = words[2]
                longitude = words[3]
                pv_type = words[5]
                capacity = words[6]
                series_name = f'{pv_type}{capacity}_{hash(latitude + longitude)}'

                df = pl.read_csv(file).with_columns(pl.col("LocalTime").str.to_datetime("%m/%d/%y %H:%M").dt.timestamp("ms")).rename({'LocalTime': 'Time', 'Power(MW)': series_name})
                master_frame = df if master_frame.is_empty() else master_frame.join(df, on='Time')

            dprint('Done.')
            return master_frame


        fetch_pv_dataset_archive()
        extract_solar_files()
        df = read_solar_data()
        cols = df.columns
        cols.remove('Time')
        return df, cols

    solar_power_df, solar_power_series = _()
    solar_power_df
    return solar_power_df, solar_power_series


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Alibaba

    https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md#introduction-of-trace-data
    """)
    return


@app.cell(hide_code=True)
def _():
    def _():
        archive = Path('data/Node_0.tar.gz')
        source_csv = Path('data/Node_0.csv')

        def fetch_alibaba_dataset_archive():
            if archive.is_file():
                dprint('Alibaba archive has already been downloaded, skipping fetching.')
                return
            dprint('Downloading archive...')
            url = 'http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/node/Node_0.tar.gz'
            r = requests.get(url)
            with open(archive, 'wb') as outfile:
                outfile.write(r.content)
            dprint('Done downloading archive.')

        def extract_alibaba_files():
            if source_csv.is_file():
                # No need to do anything
                dprint('Archive has already been unzipped, skipping decompression.')
                return
            else:
                dprint('Unzipping archive...')
                with tarfile.open(archive, 'r') as tar:
                    tar.extractall('data/', members=['Node_0.csv'], filter='data')
                    dprint('Done unzipping.')

        def read_alibaba_data() -> pl.DataFrame:
            dprint('Loading data into a DataFrame...')

            # Value to retain
            observed_value = 'node_cpu_usage' # can also be 'node_memory_usage'

            # Initial dataframe
            df = pl.read_csv(
                source_csv
            ).rename(
                {'timestamp': 'Time'}
            ).select(
                ['Time', 'nodeid', observed_value]
            ).pivot(
                on='nodeid',
                index='Time',
                values=observed_value,
                aggregate_function='first'
            ).sort(
                by='Time'
            )

            # Filter out series with null values
            null_series = pl.Series(df.null_count().unpivot(index='Time').filter(pl.col('value') != 0).select(pl.col('variable'))).to_numpy()
            dprint('Done.')
            df = df.drop(null_series)

            # Retain only first columns for now
            return df
            cols = df.columns[:2001]
            return df[cols]

        fetch_alibaba_dataset_archive()
        extract_alibaba_files()
        df = read_alibaba_data()
        cols = df.columns
        cols.remove('Time')
        return df, cols

    alibaba_df, alibaba_series = _()
    alibaba_df
    return alibaba_df, alibaba_series


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Results

    This notebook executes benchmarks on datasets, potentially loading gigabytes of data into RAM.<br/>
    To avoid filling the runtime memory, benchmarks are not executed automatically, as are the other cells of the notebook.<br/>
    You have to explicitely start them by clicking the related buttons.
    """)
    return


@app.function(hide_code=True)
# Display compressed sizes in a human-readable manner
# https://stackoverflow.com/a/43690506
def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


@app.function(hide_code=True)
# Load models with data to compare their sizes
def bench_dimensional_models(models: [FLInD], frame: pl.DataFrame, series_labels: [str], errors: [float], compute_all_series_errors: bool = False) -> [pl.DataFrame]:
    # Prepare to receive results
    model_names = []
    model_sizes = []
    models_seg_count = []
    model_write_times = []
    model_read_times = []
    model_compression_times = []
    # models => dimensions
    error_vectors = [[[] for i in range(len(series_labels))] for m in range(len(models))]

    frame_length = len(frame)
    subframe = frame.select(['Time'] + series_labels)

    with mo.status.progress_bar(total=len(models), remove_on_exit=True, show_eta=False, show_rate=False) as model_bar:
        for index, model in enumerate(models):
            model_bar.update_progress(0, title=f'Benchmarking model {model.name()}')
            model_names.append(model.name())
            dprint(f'Start benchmarking model {model.name()}.')

            # Load up model
            dprint(f'-> Load up model with dataset...')
            e = errors[0] if len(errors) == 1 else errors[index]
            model.setError(e)
            t = time.time()
            model.add_frame(frame, series_labels)
            model_write_times.append((time.time() - t) / frame_length)

            dprint(f'-> Compare model data with raw data...')
            vecs = [[] for i in range(len(series_labels))]
            with mo.status.progress_bar(total=frame_length, subtitle='Comparing model data with raw data...', remove_on_exit=True) as data_bar:
                read_time = 0
                for datum in subframe.iter_rows():
                    data_bar.update()
                    t = time.time()
                    p_time = datum[0]
                    payload = datum[1:]
                    model_value = model.read(p_time)
                    read_time += time.time() - t

                    if model.is_lossless():
                        continue

                    assert len(payload) == len(model_value)
                    for v_index, value in enumerate(payload):
                        vecs[v_index].append(abs(model_value[v_index] - payload[v_index]))
                error_vectors[index] = vecs
                model_read_times.append(read_time / frame_length)

            if model.requires_batch_compression():
                t = time.time()
                with mo.status.spinner(subtitle='Batch compressing...', remove_on_exit=True):
                    dprint(f'-> Trigger batch compression...')
                    model.batch_compress()
                model_compression_times.append(time.time() - t)
            else:
                model_compression_times.append(0)

            # Evaluate error
            model_sizes.append(model.size())
            models_seg_count.append(len(model.data()))

            dprint(f'-> Done benchmarking model.')
            print()
            model_bar.update()

    # Attempt to free some memory
    del subframe
    gc.collect()

    # Build DataFrame instance for main results
    stats = {}
    stats["Name"] = model_names
    stats["Error"] = errors
    stats["Size"] = model_sizes
    #stats["Models#"] = models_seg_count
    stats["Read time"] = model_read_times
    stats["Write time"] = model_write_times
    stats["Compression time"] = model_compression_times

    # Build one DataFrame instance per modeled dimension
    results = {}

    dprint(f'Compute statistics...')
    eval_metrics = ['avg', 'min', 'max', 'p05', 'q1', 'median', 'q3', 'p95', 'mse']
    total_length = sum([len(x) for x in error_vectors])
    with mo.status.progress_bar(total=total_length, title='Computing statistics...', remove_on_exit=True) as stats_bar:
        for m_index, v in enumerate(error_vectors):
            model = models[m_index]

            for c_index, d in enumerate(v):
                stats_bar.update()
                dimension = series_labels[c_index]

                if dimension not in results:
                    results[dimension] = {}
                if 'Name' not in results[dimension]:
                    results[dimension]['Name'] = model_names

                # Skip evaluation for lossless compression models
                if model.is_lossless():
                    for metric in eval_metrics:
                        if metric not in results[dimension]:
                            results[dimension][metric] = []
                        results[dimension][metric].append(0)
                    continue

                # Basics
                if 'avg' not in results[dimension]:
                    results[dimension]['avg'] = []
                results[dimension]['avg'].append(mean(d) if not model.is_lossless() else -1)

                if 'min' not in results[dimension]:
                    results[dimension]['min'] = []
                results[dimension]['min'].append(min(d))

                if 'max' not in results[dimension]:
                    results[dimension]['max'] = []
                results[dimension]['max'].append(max(d))

                # Quantiles
                s = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
                if 'p05' not in results[dimension]:
                    results[dimension]['p05'] = []
                results[dimension]['p05'].append(s[0])
                if 'q1' not in results[dimension]:
                    results[dimension]['q1'] = []
                results[dimension]['q1'].append(s[1])
                if 'median' not in results[dimension]:
                    results[dimension]['median'] = []
                results[dimension]['median'].append(s[2])
                if 'q3' not in results[dimension]:
                    results[dimension]['q3'] = []
                results[dimension]['q3'].append(s[3])
                if 'p95' not in results[dimension]:
                    results[dimension]['p95'] = []
                results[dimension]['p95'].append(s[4])

                # MSE
                if 'mse' not in results[dimension]:
                    results[dimension]['mse'] = []
                mse = mean([e*e for e in d])
                results[dimension]['mse'].append(mse)

    # Early return
    if compute_all_series_errors == False:
        dprint(f'Done.')
        return [pl.DataFrame(stats, strict=False)]

    # Compute metrics over all series error
    q1 = []
    q3 = []
    p95 = []
    p99 = []
    med = []
    lvar = []
    stdev = []
    mse = []
    for index, model in enumerate(models):
        if model.is_lossless():
            q1.append(0)
            med.append(0)
            q3.append(0)
            p95.append(0)
            p99.append(0)
            lvar.append(0)
            stdev.append(0)
            mse.append(0)
            continue
        all_model_errors = []
        for v in error_vectors[index]:
            all_model_errors += v
        m = np.quantile(all_model_errors, [0.25, 0.5, 0.75, 0.95, 0.99])
        q1.append(m[0])
        med.append(m[1])
        q3.append(m[2])
        p95.append(m[3])
        p99.append(m[4])
        mse.append(mean([e*e for e in all_model_errors]))
        lvar.append(var(all_model_errors))
        stdev.append(std(all_model_errors))
    #stats['Q1'] = q1
    #stats['Median'] = med
    stats['Q3'] = q3
    stats['P95'] = p95
    stats['P99'] = p99
    stats['MSE'] = mse
    stats['Variance'] = lvar
    stats['St.dev.'] = stdev

    # Build frames array
    frames = [pl.DataFrame(stats, strict=False)]
    for key in results:
        frames.append(pl.DataFrame(results[key], strict=False))

    dprint(f'Done.')
    return frames


@app.function(hide_code=True)
# Display human-readable statistics over benchmarked models
def format_dimensional_bench_results(frames: [pl.DataFrame], series_labels: [str], display_error_stats: bool = False):
    """
    Formats output of the {bench_dimensional_models} method.

    Assuming {bench_dimensional_models} returns an array of DataFrames {frame}
    which first frame contains metastatistics (statistics computed against all
    series), this collects data from first frame and displays it in a table.

    If {display_error_stats} is true, thist collects data for all {frame}
    members (one per series), and displays statistics in an accordion component.
    Mind this can be slow when working with many series.
    """

    dimensions_count = len(series_labels)
    components = [mo.ui.table(frames[0], selection=None)]
    accordion_frames = {}

    if display_error_stats:
        # Metastatistics
        meta_names = []
        meta_avgs = []
        meta_meds = []
        for frame in frames[1:]:
            for row in frame.rows(named=True):
                meta_names.append(row['Name'])
                meta_avgs.append(row['avg'])
                meta_meds.append(row['median'])
        meta = {
            'Name': meta_names,
            'Average error': meta_avgs,
            'Median error': meta_meds
        }

        # Per series statistics
        components.append(mo.md("Error statistics:"))
        for i in range(1, len(frames)):
            accordion_frames[f'- {series_labels[i-1]}'] = mo.ui.table(frames[i], selection=None, show_download=False)

        components.append(mo.accordion({
            'Meta:': pl.DataFrame(meta),
            'Per series:': mo.accordion(accordion_frames, multiple=True)
        }))

    return mo.vstack(components, align='start')


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hardware performance profile
    """)
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    hardware_profile_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        For instance, it takes about **2 minutes** to run using an i7-8650U CPU.
        '''),
        mo.vstack([
            mo.md('Click the button to start the hardware profile benchmark.'),
            hardware_profile_stop_button
        ]).callout(kind='warn')
    ])
    return (hardware_profile_stop_button,)


@app.cell(hide_code=True)
def _(hardware_profile_df, hardware_profile_stop_button):
    def _():
        mo.stop(not hardware_profile_stop_button.value)
        sample_size = 200
        model_error = 0 #computeStdErrorsArray(hardware_profile_df, 128)
        return get_clustering_steps(hardware_profile_df, hardware_profile_columns, sample_size, model_error, True, [2, 1.8])

    hardware_profile_columns = ['Fan Control #1', 'Fan Control #2', 'Fan Control #3', 'Fan Control #4', 'Fan Control #5', 'Fan Control #6', 'CPU VCore', 'Voltage #2', 'AVCC', '3VCC', 'Voltage #5', 'Voltage #6', 'Voltage #7', '3VSB', 'VBAT', 'VTT', 'Voltage #12', 'Voltage #13', 'Voltage #14', 'Voltage #15', 'CPU Core', 'Temperature #1', 'Temperature #2', 'Temperature #3', 'Temperature #5', 'Temperature #6', 'Fan #1', 'Fan #2', 'Fan #3', 'Fan #4', 'Fan #5', 'CPU Core #1', 'CPU Core #2', 'CPU Core #3', 'CPU Core #4', 'CPU Total', 'CPU Core #1_duplicated_0', 'CPU Core #2_duplicated_0', 'CPU Core #3_duplicated_0', 'CPU Core #4_duplicated_0', 'CPU Package', 'CPU Core #1_duplicated_1', 'CPU Core #2_duplicated_1', 'CPU Core #3_duplicated_1', 'CPU Core #4_duplicated_1', 'CPU Package_duplicated_0', 'CPU Cores', 'CPU Graphics', 'CPU DRAM', 'Bus Speed', 'Memory', 'Used Memory', 'Available Memory', 'GPU Core', 'GPU Fan', 'GPU Fan_duplicated_0', 'GPU Core_duplicated_0', 'GPU Memory', 'GPU Core_duplicated_1', 'GPU Core_duplicated_2', 'Temperature', 'Total LBA Written', 'Used Space', 'Temperature_duplicated_0', 'Used Space_duplicated_0', 'Used Space_duplicated_1']
    hardware_profile_cliques, hardware_profile_carousel = _()
    hardware_profile_carousel
    return hardware_profile_cliques, hardware_profile_columns


@app.cell(hide_code=True)
def _(hardware_profile_cliques, hardware_profile_columns, hardware_profile_df):
    def _():
        # Model all series individually, and with FLInD
        l = len(hardware_profile_columns)
        frames = bench_dimensional_models([RawStorage(l), FLInD(l), FLInDwFLI(l)], hardware_profile_df, hardware_profile_columns, [0, 0, 0])
        result = frames[0]
        raw_size = result['Size'][0]
        all_flind_size = result['Size'][1]
        all_fli_size = result['Size'][2]


        # Model the cluster with FLInD and the rest individually
        # (should be better than first stats)

        ## FLInD-modelling of the biggest found cluster
        labels = list(max(hardware_profile_cliques, key=len))
        l = len(labels)
        frames = bench_dimensional_models([FLInD(l)], hardware_profile_df, labels, [0])
        flind_part_size = frames[0]['Size'][0]

        ## Separated modelling of other series
        other_series = l3 = [series for series in hardware_profile_columns if series not in labels]
        l = len(other_series)
        frames = bench_dimensional_models([FLInDwFLI(l)], hardware_profile_df, other_series, [0])
        separated_part_size = frames[0]['Size'][0]


        # Layout
        return mo.md(f"""
        Here are the size results of modelling all series of the hardware profile dataset:

        * Raw size: **{human_readable_size(raw_size)}**;
        * Modelling each series separately: **{human_readable_size(all_fli_size)}**;
        * Modelling all series with FLInD: **{human_readable_size(all_flind_size)}**;
        * Modelling biggest clique series with FLInD, and other series separately: **{human_readable_size(flind_part_size + separated_part_size)}**.
        """)
    _()
    return


@app.cell(hide_code=True)
def _(hardware_profile_cliques, hardware_profile_df):
    def _():
        # Store modeled series to avoid using them multiple times
        modeled_series = set()
        total_size = 0

        # Sort cliques by size
        cliques = sorted(hardware_profile_cliques, key=len, reverse=True)

        for index, clique in enumerate(cliques):
            # Filter out series already modeled in previous cliques
            series_to_model = [series for series in clique if series not in modeled_series]
            modeled_series.update(series_to_model)
            l = len(series_to_model)
            dprint(f'For clique #{index}, {'no series to model' if l == 0 else f'modelling {l} series (out of {len(clique)})'}.')

            # No need to load a model if there's no series to model
            if l == 0:
                continue

            # Model clique series
            m = FLInD(l)
            m.setError(0)
            m.add_frame(hardware_profile_df, series_to_model)
            total_size += m.size()

        return mo.md(f'''
        These results are obtained using the biggest clique only.<br/>
        The size when splitting series between multiple models (using all cliques) is **{human_readable_size(total_size)}**.
        ''')
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What if we try to use all cliques containing more than one series?
    """)
    return


@app.cell(hide_code=True)
def _(hardware_profile_cliques, hardware_profile_df):
    def _():
        # Store modeled series to avoid using them multiple times
        modeled_series = set()
        total_size = 0

        # Series to be stored individually
        separated_series = set()

        # Sort cliques by size
        cliques = sorted(hardware_profile_cliques, key=len, reverse=True)
        final_clique_sizes = []

        for index, clique in enumerate(cliques):
            # Filter out series already modeled in previous cliques
            series_to_model = [series for series in clique if series not in modeled_series]
            modeled_series.update(series_to_model)
            l = len(series_to_model)
            dprint(f'For clique #{index}, {'no series to model' if l == 0 else f'modelling {l} series (out of {len(clique)})'}.')

            # No need to load a model if there's no series to model
            if l == 0:
                continue

            # If clique contains one series only, model it separately
            if l == 1:
                separated_series.add(series_to_model[0])
                continue

            # Model clique series
            m = FLInD(l)
            m.setError(0)
            m.add_frame(hardware_profile_df, series_to_model)
            total_size += m.size()
            final_clique_sizes.append(l)

        # Model separated series
        n = len(separated_series)
        m = FLInDwFLI(n)
        m.setError(0)
        m.add_frame(hardware_profile_df, list(separated_series))
        total_size += m.size()

        return mo.md(f'''
        Modelling all cliques containing more than one series results in {len(final_clique_sizes)} combined models with sizes: {final_clique_sizes}<br/>and one separated model containing {n} series, with a total cost of **{human_readable_size(total_size)}**.
        ''')
    _()
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    hardware_profile_bench_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        It benchmarks models several times, then average statistics.
        '''),
        mo.vstack([
            mo.md('Click the button to start the hardware profile benchmark.'),
            hardware_profile_bench_stop_button
        ]).callout(kind='warn')
    ])
    return (hardware_profile_bench_stop_button,)


@app.cell(hide_code=True)
def _(
    TDengine,
    hardware_profile_bench_stop_button,
    hardware_profile_columns,
    hardware_profile_df,
):
    def _():
        mo.stop(not hardware_profile_bench_stop_button.value)
        dimensions_count = len(hardware_profile_columns)

        iterations_count = 5
        names = []
        sizes = []
        read_times = []
        write_times = []
        compression_times = []

        for i in range(iterations_count):
            models = [
                # ZipStorage(dimensions_count),
                # RawStorage(dimensions_count),
                # FLInDwFLI(dimensions_count),
                # FLInD(dimensions_count),
                # CliqueFLInD(dimensions_count),
                # TimescaleDB(hardware_profile_columns),
                TDengine(hardware_profile_columns)
            ]
            errors = [0 for i in range(len(models))]
            frame = bench_dimensional_models(models, hardware_profile_df, hardware_profile_columns, errors)[0]
            read_times.append(np.array(frame["Read time"]))
            write_times.append(np.array(frame["Write time"]))
            compression_times.append(np.array(frame["Compression time"]))

            if i == 0:
                names = frame['Name']
                sizes = frame['Size']

        read_averages = [np.mean(x) for x in zip(*read_times)]
        write_averages = [np.mean(x) for x in zip(*write_times)]
        compression_averages = [np.mean(x) for x in zip(*compression_times)]
        read_averages = ["{0:.3e}".format(x) for x in read_averages]
        write_averages = ["{0:.3e}".format(x) for x in write_averages]
        compression_averages = ["{0:.3e}".format(x) for x in compression_averages]

        # Convert sizes
        dprint(sizes)
        sizes = [human_readable_size(s) for s in sizes]

        return pl.DataFrame({
            'Name': names,
            'Size': sizes,
            'Read time': read_averages,
            'Write time': write_averages,
            'Compression time': compression_averages
        })

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Electricity consumption
    """)
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    electricity_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        For instance, it takes about **17 minutes** to run using an i7-8650U CPU.
        '''),
        mo.vstack([
            mo.md('Click the button to start the electricity consumption benchmark.'),
            electricity_stop_button
        ]).callout(kind='warn')
    ])
    return (electricity_stop_button,)


@app.cell(hide_code=True)
def _(electricity_df, electricity_series, electricity_stop_button):
    def _():
        mo.stop(not electricity_stop_button.value)
        sample_size = 200
        model_error = 0
        return get_clustering_steps(electricity_df, electricity_series, sample_size, model_error)

    electricity_consumption_cliques, electricity_consumption_carousel = _()
    electricity_consumption_carousel
    return (electricity_consumption_cliques,)


@app.cell(hide_code=True)
def _(electricity_consumption_cliques, electricity_df, electricity_series):
    def _():
        # Model all series individually, and with FLInD
        l = len(electricity_series)
        frames = bench_dimensional_models([FLInD(l), FLInDwFLI(l)], electricity_df, electricity_series, [0, 0])
        result = frames[0]
        all_flind_size = result['Size'][0]
        all_fli_size = result['Size'][1]


        # Model the cluster with FLInD and the rest individually
        # (should be better than first stats)

        ## FLInD-modelling of the biggest found cluster
        labels = list(max(electricity_consumption_cliques, key=len))
        l = len(labels)
        frames = bench_dimensional_models([FLInD(l)], electricity_df, labels, [0])
        flind_part_size = frames[0]['Size'][0]

        ## Separated modelling of other series
        other_series = l3 = [series for series in electricity_series if series not in labels]
        l = len(other_series)
        frames = bench_dimensional_models([FLInDwFLI(l)], electricity_df, other_series, [0])
        separated_part_size = frames[0]['Size'][0]


        # Layout
        return mo.md(f"""
        Here are the size results of modelling all series of the electricity consumption dataset:

        * Modelling each series separately: **{human_readable_size(all_fli_size)}**;
        * Modelling all series with FLInD: **{human_readable_size(all_flind_size)}**;
        * Modelling biggest clique series with FLInD, and other series separately: **{human_readable_size(flind_part_size + separated_part_size)}**.
        """)
    _()
    return


@app.cell(hide_code=True)
def _(electricity_consumption_cliques, electricity_df):
    def _():
        # Store modeled series to avoid using them multiple times
        modeled_series = set()
        total_size = 0

        # Sort cliques by size
        cliques = sorted(electricity_consumption_cliques, key=len, reverse=True)

        for index, clique in enumerate(cliques):
            # Filter out series already modeled in previous cliques
            series_to_model = [series for series in clique if series not in modeled_series]
            modeled_series.update(series_to_model)
            l = len(series_to_model)
            dprint(f'For clique #{index}, {'no series to model' if l == 0 else f'modelling {l} series (out of {len(clique)})'}.')

            # No need to load a model if there's no series to model
            if l == 0:
                continue

            # Model clique series
            m = FLInD(l)
            m.setError(0)
            m.add_frame(electricity_df, series_to_model)
            total_size += m.size()

        return mo.md(f'''
        These results are obtained using the biggest clique only.<br/>
        The size when splitting series between multiple models (using all cliques) is **{human_readable_size(total_size)}**.
        ''')
    _()
    return


@app.cell(hide_code=True)
def _(electricity_df, electricity_series, electricity_stop_button):
    def _():
        mo.stop(not electricity_stop_button.value)
        dimensions_count = len(electricity_series)
        frames = bench_dimensional_models([RawStorage(dimensions_count), FLInDwFLI(dimensions_count), CliqueFLInD(dimensions_count)], electricity_df, electricity_series, [0, 0, 0])
        return format_dimensional_bench_results(frames, electricity_series)
    _()
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    electricity_bench_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        It benchmarks models several times, then average statistics.
        '''),
        mo.vstack([
            mo.md('Click the button to start the electricity consumption benchmark.'),
            electricity_bench_stop_button
        ]).callout(kind='warn')
    ])
    return (electricity_bench_stop_button,)


@app.cell(hide_code=True)
def _(electricity_bench_stop_button, electricity_df, electricity_series):
    def _():
        mo.stop(not electricity_bench_stop_button.value)
        dimensions_count = len(electricity_series)

        iterations_count = 1
        names = []
        sizes = []
        read_times = []
        write_times = []
        compression_times = []

        for i in range(iterations_count):
            dprint(f'Iteration #{i}')
            models = [
                # RawStorage(dimensions_count),
                # FLInDwFLI(dimensions_count),
                # FLInD(dimensions_count),
                # #CliqueFLInD(dimensions_count),
                # CliqueFLInD(dimensions_count, 140256)
                TimescaleDB(electricity_series)
            ]
            errors = [0 for i in range(len(models))]
            frame = bench_dimensional_models(models, electricity_df, electricity_series, errors)[0]
            read_times.append(np.array(frame["Read time"]))
            write_times.append(np.array(frame["Write time"]))
            compression_times.append(np.array(frame["Compression time"]))

            if i == 0:
                names = frame['Name']
                sizes = frame['Size']

        read_averages = [np.mean(x) for x in zip(*read_times)]
        write_averages = [np.mean(x) for x in zip(*write_times)]
        compression_averages = [np.mean(x) for x in zip(*compression_times)]
        read_averages = ["{0:.3e}".format(x) for x in read_averages]
        write_averages = ["{0:.3e}".format(x) for x in write_averages]
        compression_averages = ["{0:.3e}".format(x) for x in compression_averages]

        # Convert sizes
        sizes = [human_readable_size(s) for s in sizes]

        return pl.DataFrame({
            'Name': names,
            'Size': sizes,
            'Read time': read_averages,
            'Write time': write_averages,
            'Compression time': compression_averages
        })

    _()
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    electricity_error_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        It explores how setting the model error impacts model performances.
        '''),
        mo.vstack([
            mo.md('Click the button to start the electricity consumption varying error benchmark.'),
            electricity_error_stop_button
        ]).callout(kind='warn')
    ])
    return (electricity_error_stop_button,)


@app.cell(hide_code=True)
def _(electricity_df, electricity_error_stop_button, electricity_series):
    def _():
        mo.stop(not electricity_error_stop_button.value)
        dimensions_count = len(electricity_series)

        models = [
            RawStorage(dimensions_count),
            CliqueFLInD(dimensions_count),
            CliqueFLInD(dimensions_count),
            CliqueFLInD(dimensions_count)
        ]
        frames = bench_dimensional_models(models, electricity_df, electricity_series, [0, 1, 2, 5])
        return format_dimensional_bench_results(frames, electricity_series)
    _()
    return


@app.cell(hide_code=True)
def _(computeStdErrorsArray, electricity_df, electricity_series):
    # Check the evolution of size while increasing tolerated size
    def _():
        return
        dimensions_count = len(electricity_series)
        models = [
            #RawStorage(dimensions_count),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 1)),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 2)),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 5)),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 10)),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 20)),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 50)),
            FLInD(dimensions_count, computeStdErrorsArray(electricity_df, 100)),
        ]
        errors = [1, 2, 5, 10, 20, 50, 100]
        assert len(models) == len(errors)

        frames = bench_dimensional_models(models, electricity_df, electricity_series, errors)
        return format_dimensional_bench_results(frames, electricity_series)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Solar power data
    """)
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    solar_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        For instance, it takes about **12 minutes** to run using an i7-8650U CPU.
        '''),
        mo.vstack([
            mo.md('Click the button to start the solar power benchmark.'),
            solar_stop_button
        ]).callout(kind='warn')
    ])
    return (solar_stop_button,)


@app.cell(hide_code=True)
def _(solar_power_df, solar_power_series, solar_stop_button):
    def _():
        mo.stop(not solar_stop_button.value)
        sample_size = 200
        model_error = 0
        return get_clustering_steps(solar_power_df, solar_power_series, sample_size, model_error)

    solar_power_cliques, solar_power_carousel = _()
    solar_power_carousel
    return (solar_power_cliques,)


@app.cell(hide_code=True)
def _(solar_power_cliques, solar_power_df, solar_power_series):
    def _():
        # Model all series individually, and with FLInD
        l = len(solar_power_series)
        frames = bench_dimensional_models([FLInD(l), FLInDwFLI(l)], solar_power_df, solar_power_series, [0, 0])
        result = frames[0]
        all_flind_size = result['Size'][0]
        all_fli_size = result['Size'][1]


        # Model the cluster with FLInD and the rest individually
        # (should be better than first stats)

        ## FLInD-modelling of the biggest found cluster
        labels = list(max(solar_power_cliques, key=len))
        l = len(labels)
        frames = bench_dimensional_models([FLInD(l)], solar_power_df, labels, [0])
        flind_part_size = frames[0]['Size'][0]

        ## Separated modelling of other series
        other_series = l3 = [series for series in solar_power_series if series not in labels]
        l = len(other_series)
        frames = bench_dimensional_models([FLInDwFLI(l)], solar_power_df, other_series, [0])
        separated_part_size = frames[0]['Size'][0]


        # Layout
        return mo.md(f"""
        Here are the size results of modelling all series of the solar power dataset:

        * Modelling each series separately: **{all_fli_size} floats**;
        * Modelling all series with FLInD: **{all_flind_size} floats**;
        * Modelling biggest clique series with FLInD, and other series separately: **{flind_part_size + separated_part_size} floats**.
        """)
    _()
    return


@app.cell(hide_code=True)
def _(solar_power_df, solar_power_series, solar_stop_button):
    def _():
        mo.stop(not solar_stop_button.value)
        dimensions_count = len(solar_power_series)
        frames = bench_dimensional_models([RawStorage(dimensions_count), FLInDwFLI(dimensions_count), CliqueFLInD(dimensions_count)], solar_power_df, solar_power_series, [0, 0, 0])
        return format_dimensional_bench_results(frames, solar_power_series)
    _()
    return


@app.cell
def _():
    epsilon_solar_drift_slider = mo.ui.slider(0, 1, 0.0001, label='Percentile', debounce=True, show_value=True)
    return (epsilon_solar_drift_slider,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Understanding FLInD gain
    """)
    return


@app.function(hide_code=True)
def computeSeriesDrifts(time: pl.Series, series: pl.Series, errorThreshold: float) -> float:
    drifts = []
    for i in range(len(time) - 1):
        drift = abs((series[i+1] - series[i]) / (time[i+1] - time[i]))
        drifts.append(drift)
    drifts.sort()
    return drifts


@app.cell(hide_code=True)
def _(epsilon_solar_drift_slider, solar_power_df, solar_power_series):
    # Checking dataset deltas
    def _():
        percentile = epsilon_solar_drift_slider.value
        study_col = solar_power_series[2]
        x = computeSeriesDrifts(solar_power_df['Time'], solar_power_df[study_col], percentile)

        reset_plt_font_scale()
        N = len(x)
        y = np.arange(N) / float(N)
        plt.plot(x, y, label="CPU usage")

        x_percentile = np.percentile(x, percentile * 100)
        plt.vlines(x=x_percentile, ymin=0, ymax=percentile, color='r', linestyles=':', linewidth=1)
        plt.hlines(y=percentile, xmin=0, xmax=x_percentile, color='r', linestyles=':', linewidth=1)
        plt.yticks([0, 0.5, 1, percentile])
        text_x = x_percentile + 0.0000001
        plt.text(text_x, percentile-0.03, "Drift: {:.2e}".format(x_percentile), c="red", fontsize=7)

        plt.title(f'CDF of "{study_col}" series point-to-point deltas')
        plt.xlabel('Delta value')
        plt.ylabel('$\mathbb{P}(N<x)$')

        return mo.vstack([
            plt.gca(),
            epsilon_solar_drift_slider
        ], align='center')
    _()
    return


@app.function(hide_code=True)
# Check all series deltas of a given dataset
def compute_dataset_drifts(dataset, series, percentile):
    results = {}

    with mo.status.progress_bar(title='Computing drifts over all series', total=len(series)) as bar:
        for s in series:
            x = computeSeriesDrifts(dataset['Time'], dataset[s], percentile)
            x_percentile = np.percentile(x, percentile * 100)

            if x_percentile not in results:
                results[x_percentile] = 1
            else:
                results[x_percentile] += 1

            bar.update()

    dprint(f'Computing the drift value of {len(series)} series with p={percentile} results in:')
    for p_value in results.keys():
        dprint(f'-> {results[p_value]} series having a drift value of {p_value}.')


@app.cell(hide_code=True)
def _():
    #compute_dataset_drifts(solar_power_df, solar_power_series, 0.53)
    return


@app.cell(hide_code=True)
def _(solar_power_df, solar_power_series):
    # Big benchmark (not run by default since it takes so much time)
    def _():
        return
        dimensions_count = len(solar_power_series)

        iterations_count = 10
        names = []
        sizes = []
        read_times = []
        write_times = []
        compression_times = []

        for i in range(iterations_count):
            models = [
                # RawStorage(dimensions_count),
                # FLInDwFLI(dimensions_count),
                # FLInD(dimensions_count),
                # CliqueFLInD(dimensions_count),
                # CliqueFLInD(dimensions_count, 105120),
                TimescaleDB(solar_power_series)
            ]
            errors = [0 for i in range(len(models))]
            frame = bench_dimensional_models(models, solar_power_df, solar_power_series, errors)[0]
            read_times.append(np.array(frame["Read time"]))
            write_times.append(np.array(frame["Write time"]))
            compression_times.append(np.array(frame["Compression time"]))

            if i == 0:
                names = frame['Name']
                sizes = frame['Size']

        read_averages = [np.mean(x) for x in zip(*read_times)]
        write_averages = [np.mean(x) for x in zip(*write_times)]
        compression_averages = [np.mean(x) for x in zip(*compression_times)]
        read_averages = ["{0:.3e}".format(x) for x in read_averages]
        write_averages = ["{0:.3e}".format(x) for x in write_averages]
        compression_averages = ["{0:.3e}".format(x) for x in compression_averages]

        # Convert sizes
        sizes = [human_readable_size(s) for s in sizes]

        return pl.DataFrame({
            'Name': names,
            'Size': sizes,
            'Read time': read_averages,
            'Write time': write_averages,
            'Compression time': compression_averages
        })

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alibaba
    """)
    return


@app.cell(hide_code=True)
def _():
    # Don't run the benchmark automatically, as it uses lots of resources
    alibaba_stop_button = mo.ui.run_button(kind='warn', label='Run benchmark')

    mo.vstack([
        mo.md('This benchmark is not executed automatically, as it uses big amounts of memory.'),
        mo.vstack([
            mo.md('Click the button to start alibaba benchmarks.'),
            alibaba_stop_button
        ]).callout(kind='warn')
    ])
    return (alibaba_stop_button,)


@app.cell(hide_code=True)
def _(alibaba_df, alibaba_series, alibaba_stop_button):
    alibaba_sample_size = 1440
    alibaba_model_error = 0.0
    alibaba_threshold = 0.1

    def _():
        mo.stop(not alibaba_stop_button.value)
        return get_clustering_steps(alibaba_df, alibaba_series, alibaba_sample_size, alibaba_model_error, True, [2, 1.8], alibaba_threshold)

    alibaba_cliques, alibaba_carousel = _()
    alibaba_carousel
    return alibaba_cliques, alibaba_model_error


@app.cell(hide_code=True)
def _(alibaba_cliques, alibaba_df, alibaba_model_error, alibaba_series):
    def _():
        # Model all series individually, and with FLInD
        l = len(alibaba_series)
        models = [RawStorage(l), FLInD(l), FLInDwFLI(l)]
        errors = [0, alibaba_model_error, alibaba_model_error]
        assert len(errors) == len(models)
        frames = bench_dimensional_models(models, alibaba_df, alibaba_series, errors)
        result = frames[0]
        raw_size = result['Size'][0]
        all_flind_size = result['Size'][1]
        all_fli_size = result['Size'][2]

        result_string = f'Here are the size results of modelling all series of the alibaba dataset:<ul><li>Raw size: **{raw_size} floats**;</li><li>Modelling each series separately: **{all_fli_size} floats**;</li><li>Modelling all series with FLInD: **{all_flind_size} floats**'

        # Retrieve cliques, and return if there's only one
        if len(alibaba_cliques) == 1:
            print()
            dprint("Only one clique found, no need to do separate modelling.")
            result_string += '.</li></ul>Only one clique was found, so there is no need to do separate modelling.'
            return mo.md(result_string)


        result_string += ';</li>'
        # Model biggest clique with FLInD and the other series individually
        ## FLInD-modelling of the biggest found cluster
        labels = list(max(alibaba_cliques, key=len))
        l = len(labels)
        frames = bench_dimensional_models([FLInD(l)], alibaba_df, labels, [alibaba_model_error])
        flind_part_size = frames[0]['Size'][0]

        ## Separated modelling of other series
        other_series = l3 = [series for series in alibaba_series if series not in labels]
        l = len(other_series)
        frames = bench_dimensional_models([FLInDwFLI(l)], alibaba_df, other_series, [alibaba_model_error])
        separated_part_size = frames[0]['Size'][0]


        # Layout
        result_string += f'<li>Modelling biggest clique series with FLInD, and other series separately: **{flind_part_size + separated_part_size} floats**.</li></ul>'
        return mo.md(result_string)
    _()
    return


@app.cell(hide_code=True)
def _(alibaba_cliques, alibaba_df):
    def _():
        # Store modeled series to avoid using them multiple times
        modeled_series = set()
        total_size = 0

        # Sort cliques by size
        cliques = sorted(alibaba_cliques, key=len, reverse=True)

        for index, clique in enumerate(cliques):
            # Filter out series already modeled in previous cliques
            series_to_model = [series for series in clique if series not in modeled_series]
            modeled_series.update(series_to_model)
            l = len(series_to_model)
            dprint(f'For clique #{index}, {'no series to model' if l == 0 else f'modelling {l} series (out of {len(clique)})'}.')

            # No need to load a model if there's no series to model
            if l == 0:
                continue

            # Model clique series
            m = FLInD(l)
            m.setError(0)
            m.add_frame(alibaba_df, series_to_model)
            total_size += m.size()

        return mo.md(f'''
        These results are obtained using the biggest clique only.<br/>
        The size when splitting series between multiple models (using all cliques) is **{total_size} floats**.
        ''')
    _()
    return


@app.cell(hide_code=True)
def _(alibaba_df, alibaba_series):
    # Big benchmark (not run by default since it takes so much time)
    def _():
        return
        dimensions_count = len(alibaba_series)

        iterations_count = 1
        names = []
        sizes = []
        read_times = []
        write_times = []
        compression_times = []

        for i in range(iterations_count):
            print(f'Iteration #{i}')

            models = [
                RawStorage(dimensions_count),
                FLInDwFLI(dimensions_count),
                FLInD(dimensions_count),
                CliqueFLInD(dimensions_count),
                # TimescaleDB(alibaba_series) # max column count = 1600, not enough to fit alibaba
            ]
            errors = [0 for i in range(len(models))]
            frame = bench_dimensional_models(models, alibaba_df, alibaba_series, errors)[0]
            read_times.append(np.array(frame["Read time"]))
            write_times.append(np.array(frame["Write time"]))
            compression_times.append(np.array(frame["Compression time"]))

            if i == 0:
                names = frame['Name']
                sizes = frame['Size']

        read_averages = [np.mean(x) for x in zip(*read_times)]
        write_averages = [np.mean(x) for x in zip(*write_times)]
        compression_averages = [np.mean(x) for x in zip(*compression_times)]
        read_averages = ["{0:.3e}".format(x) for x in read_averages]
        write_averages = ["{0:.3e}".format(x) for x in write_averages]
        compression_averages = ["{0:.3e}".format(x) for x in compression_averages]

        # Convert sizes
        sizes = [human_readable_size(s) for s in sizes]

        return pl.DataFrame({
            'Name': names,
            'Size': sizes,
            'Read time': read_averages,
            'Write time': write_averages,
            'Compression time': compression_averages
        })

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bruteforce correlation search

    Clique search might return one clique including all series together, indicating there is no need to separately model series.<br/>
    We empirically validate this by bruteforcing all possibilities of storing series in two models, one FLInD combined model and one separated model.

    To do so, we intentionally select highly correlated series, to ensure the clique selection approach returns only one series cluster.
    """)
    return


@app.cell
def _():
    min_pearson = 0.9
    return (min_pearson,)


@app.cell(hide_code=True)
def _(alibaba_df, alibaba_series):
    def build_correlated_alibaba_dataset(min_pearson = 0.5):
        final_series = set()
        first_series = alibaba_series[0:3]

        # Retain close series only
        for studied_series in first_series:
            studied_series_data = alibaba_df[studied_series]
            close_series = [series for series in alibaba_series if stats.pearsonr(studied_series_data, alibaba_df[series]).correlation >= min_pearson]
            final_series.update(close_series)

        series = list(final_series)
        return alibaba_df[(['Time'] + series)], series
    return (build_correlated_alibaba_dataset,)


@app.cell(hide_code=True)
def _(build_correlated_alibaba_dataset, min_pearson):
    alibaba_correlated_df, alibaba_correlated_series = build_correlated_alibaba_dataset(min_pearson)
    alibaba_correlated_df
    return alibaba_correlated_df, alibaba_correlated_series


@app.cell(hide_code=True)
def _(alibaba_correlated_df, alibaba_correlated_series):
    def _():
        l = len(alibaba_correlated_series)
        frames = bench_dimensional_models([RawStorage(l)], alibaba_correlated_df, alibaba_correlated_series, [0])
        return frames[0]
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Then, we run both bruteforce and clique selection methods to ensure both results are aligned.
    """)
    return


@app.function(hide_code=True)
def bruteforce_modelling_combinations(frame: pl.DataFrame, series: [str], error: float = 0):
    # Store bruteforce method duration
    t = time.time()

    # Store series references in a map to avoid computing products of strings (very expensive)
    index_to_series = {}
    for i, s in enumerate(series):
        index_to_series[i] = s
    series_indexes = range(len(series))

    # Create matrix of combinations
    ## https://stackoverflow.com/a/48472670
    def all_combinations(width, vals):
        return np.array(np.meshgrid(*[vals]*width, indexing='ij')).reshape((width,-1)).transpose()

    combinations_count = 2 ** len(series)
    products = all_combinations(len(series), [0, 1])

    # Storing results here
    a_flind_names = []
    a_fli2d_names = []
    a_sizes = []

    # Compute all possibilities
    with mo.status.progress_bar(total=pow(2, len(series)), completion_title="Done!", completion_subtitle=f"Check the dataframe for results.") as bar:
        for c_index, combination in enumerate(products):
            zeroes = np.where(combination == 0)[0]
            fli2d_series = [index_to_series[i] for i in zeroes]
            flind_series = [index_to_series[i] for i in series_indexes if i not in zeroes]

            bar.update(title="Benchmarking models", subtitle=f"Processing association #{c_index}...")
            #dprint(f'For association #{c_index}, modelling:\n\t- {flind_series} with FLInD;\n\t- {fli2d_series} with FLI.')

            total_size = 0

            ### FLInD
            flind_series_length = len(flind_series)
            if flind_series_length > 0:
                flind_cols = ['Time'] + flind_series
                series_frame = frame[flind_cols]

                model = FLInD(flind_series_length)
                model.setError(error)
                for row in series_frame.iter_rows():
                    model.add(row[0], row[1:])
                total_size += model.size()

            ### FLI2D
            fli2d_series_length = len(fli2d_series)
            if fli2d_series_length > 0:
                fli2d_cols = ['Time'] + fli2d_series
                series_frame = frame[fli2d_cols]

                model = FLInDwFLI(fli2d_series_length)
                model.setError(error)
                for row in series_frame.iter_rows():
                    model.add(row[0], row[1:])
                total_size += model.size()

            a_flind_names.append(f'{flind_series}')
            a_fli2d_names.append(f'{fli2d_series}')
            a_sizes.append(total_size)
            #dprint(f'Size: {total_size}')
            #print()

    # Save final duration
    t = time.time() - t

    return pl.DataFrame({
        'Combined model (FLInD)': a_flind_names,
        'Separate model (FLInDwFLI)': a_fli2d_names,
        'Total size': a_sizes
    }).sort('Total size', descending=False), t


@app.cell(hide_code=True)
def _(alibaba_correlated_series):
    # Don't run bruteforce benchmark automatically, as it is resource intensive
    correlated_alibaba_stop_button = mo.ui.run_button(kind='warn', label='Run bruteforce benchmark')

    mo.vstack([
        mo.md('''
        Bruteforce benchmark computes all the possible ways of splitting series between a FLInD, combined model and a FLInDwFLI, separate model: its runtime is therefore directly linked to the count of series, as there are $2^n$ modelling possibilities for n series.<br/>
        As an example, setting the above Pearson threshold to 0.887 results in a frame containing **18 series**; computing the **262,144 modelling possibilities** takes about **70 minutes** to run using an i7-11700 CPU.
        '''),
        mo.vstack([
            mo.md(f'Click the button to start the bruteforce benchmark on {len(alibaba_correlated_series)} series.'),
            correlated_alibaba_stop_button
        ]).callout(kind='warn')
    ])
    return (correlated_alibaba_stop_button,)


@app.cell(hide_code=True)
def _(alibaba_correlated_series, alibaba_df, correlated_alibaba_stop_button):
    def _():
        mo.stop(not correlated_alibaba_stop_button.value)
        return bruteforce_modelling_combinations(alibaba_df, alibaba_correlated_series)

    def _format():
        first_result = alibaba_correlated_bruteforce_results.rows()[0]
        flind_series_count = len(eval(first_result[0]))
        fli2d_series_count = len(eval(first_result[1]))
        l = len(alibaba_correlated_series)

        return mo.vstack([
            alibaba_correlated_bruteforce_results,
            mo.md(f'According to bruteforce results, the best way to store the **{l} series** requires **{first_result[2]} floats**, and consists in splitting series as follows:<ul><li>{flind_series_count}/{l} series combined together;</li><li>{fli2d_series_count}/{l} series modelled separately.</li></ul>'),
            mo.md(f'Computing all repartitions took {alibaba_correlated_bruteforce_duration} seconds.')
        ], align='stretch', justify='center')

    alibaba_correlated_bruteforce_results, alibaba_correlated_bruteforce_duration = _()
    _format()
    return


@app.cell(hide_code=True)
def _():
    alibaba_clique_sample_size = 200 # max: 1440
    alibaba_correlated_threshold = 0.9
    return alibaba_clique_sample_size, alibaba_correlated_threshold


@app.cell(hide_code=True)
def _(
    alibaba_clique_sample_size,
    alibaba_correlated_df,
    alibaba_correlated_series,
    alibaba_correlated_threshold,
    correlated_alibaba_stop_button,
):
    def _():
        mo.stop(not correlated_alibaba_stop_button.value)
        t = time.time()

        matrix = compute_timestamps_matrix(alibaba_correlated_df, alibaba_correlated_series, alibaba_clique_sample_size, 0, False)
        graph = clusters_search(matrix, alibaba_correlated_series, alibaba_correlated_threshold)
        series_groups = list(nx.algorithms.clique.find_cliques(graph))
        best_group = list(sorted(max(series_groups, key=len)))
        t = time.time() - t

        return mo.hstack([
            mo.vstack([
                mo.md('**Finding the largest clique**'),
                mo.md('We look for the biggest clique, *e.g.* the one that contains the most series to model, and display it.'),
                mo.md(f'**Here, the biggest clique encompasses {len(best_group)} series (out of {len(alibaba_correlated_series)}).**'),
                mo.md(f'Additionally, result was obtained in {t} seconds.')
            ]),# + [
            #     mo.md("\n".join([f'- {series}' for series in best_group])),
            # ]),
            display_graph(graph, best_group)
        ], widths=[1.8, 2])

    _()
    return


@app.cell(hide_code=True)
def _():
    def computeStdErrorsArray(frame: pl.DataFrame, error: float) -> [float]:
        series_means = frame.std().rows()[0]
        r = [error*series_means[i] for i in range(len(series_means))]
        # Not returning the error computed over the timestamp series
        return r[1:]

    def computeMeanErrorsArray(frame: pl.DataFrame, error: float) -> [float]:
        series_means = frame.mean().rows()[0]
        r = [(error/100)*series_means[i] for i in range(len(series_means))]
        return r[1:]
    return computeMeanErrorsArray, computeStdErrorsArray


@app.cell(hide_code=True)
def _(
    alibaba_df,
    alibaba_series,
    computeMeanErrorsArray,
    correlated_alibaba_stop_button,
):
    # Check the evolution of size while increasing tolerated size
    def _():
        mo.stop(not correlated_alibaba_stop_button.value)
        dimensions_count = len(alibaba_series)
        models = [
            RawStorage(dimensions_count),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 1)),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 2)),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 5)),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 10)),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 20)),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 50)),
            FLInDwFLI(dimensions_count, computeMeanErrorsArray(alibaba_df, 100))
        ]
        errors = [0, 1, 2, 5, 10, 20, 50, 100]
        assert len(models) == len(errors)

        frames = bench_dimensional_models(models, alibaba_df, alibaba_series, errors, True)
        return format_dimensional_bench_results(frames, alibaba_series)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Splitting series between fixed-size models
    """)
    return


@app.cell(hide_code=True)
def _(computeStdErrorsArray):
    class SplitFLInD(FLInD):
        def __init__(self, dimensions_count, max_series_count = 100, stds_count = 1):
            self.dimensions_count = dimensions_count
            self.max_series_count = max_series_count
            self.stds_count = stds_count
            self.models = []
            FLInD.__init__(self, dimensions_count)

        def _models(self):
            return self.models

        def setError(self, value: float):
            pass

        def name(self):
            return f'SplitFLInD (n={self.dimensions_count}, s={self.max_series_count})'

        def data(self):
            d = []
            for m in self._models():
                d += m.data()
            return d

        def size(self):
            total = 0
            for m in self._models():
                total += m.size()
            return total

        def add_frame(self, frame: pl.DataFrame, series: [str], sample_size: int = -1) -> None:
            limit = len(frame) if sample_size == -1 else sample_size
            subframe = frame.select(['Time'] + series).limit(limit)

            series_count = len( series )
            submodels_count = math.ceil(series_count / self.max_series_count) + 1

            for i in range(submodels_count):
                beginIndex = 1+ i*self.max_series_count
                if beginIndex > series_count:
                    break
                endIndex = 1+ (i+1)*self.max_series_count
                if endIndex > series_count:
                    endIndex = series_count + 1

                label = f'[{beginIndex}; {endIndex}]'
                dprint(label)
                dimensions_count = endIndex - beginIndex
                columns = frame.columns[beginIndex:endIndex]
                ssubframe = frame[['Time'] + columns]
                m = FLInD(dimensions_count, computeStdErrorsArray(ssubframe, self.stds_count))
                m.add_frame(ssubframe, columns)
                self.models.append(m)

        def add(self, tRaw: float, vRaw: tuple):
            raise 'Unimplemented'

        def read(self, t: float) -> tuple:
            new_point = []
            for model in self.models:
                new_point.extend( model.read(t) )
            return tuple(new_point)
    return (SplitFLInD,)


@app.cell(hide_code=True)
def _():
    # Splitting dataframe into several models
    split_alibaba_stop_button = mo.ui.run_button(kind='warn', label='Run split benchmark')

    mo.vstack([
        mo.md('''
        This benchmark is not executed automatically, as it uses big amounts of memory.<br/>
        It explores how splitting series between different models improves total storage size.
        '''),
        mo.vstack([
            mo.md('Click the button to start the split benchmark.'),
            split_alibaba_stop_button
        ]).callout(kind='warn')
    ])
    return (split_alibaba_stop_button,)


@app.function(hide_code=True)
def normalize(frame: pl.DataFrame) -> pl.DataFrame:
    columns = frame.columns[1:]
    return frame[columns].select(
        (pl.all()-pl.all().min())/(pl.all().max()-pl.all().min())
        #(pl.all() - pl.all().mean()) / pl.all().std()
    ).insert_column(
        0, pl.Series(frame['Time'])
    ).fill_nan(1)


@app.cell(hide_code=True)
def _(SplitFLInD, alibaba_df, split_alibaba_stop_button):
    def _():
        mo.stop(not split_alibaba_stop_button.value)
        frame = normalize(alibaba_df)
        series_count = len( frame.columns ) - 1 # Not taking 'Time' into account

        # Results stats
        cluster_labels = []
        raw_sizes = []
        zero_sizes = []
        std_sizes = []

        raw_total_size = 0
        zero_error_total_size = 0
        one_std_total_size = 0

        # Compare with putting all series in one model
        models = [
            #FLInD(series_count, computeStdErrorsArray(frame, 1)),
            SplitFLInD(series_count, 40)
        ]
        errors = [1 for i in range(len(models))]
        all_cols = frame.columns
        all_cols.remove('Time')
        frames = bench_dimensional_models(models, frame, all_cols, errors, True)[0]
        return frames
    _()
    return


@app.cell(hide_code=True)
def _():
    def _():
        # data
        series_packet_size = [100, 50, 20, 10, 5, 2, 1]
        sizes = [15602333, 12772368, 9077549, 6639695, 4650928, 2206167, 26104]
        read_times = [0.002862049473656548, 0.003714356323083242, 0.005750617219342126, 0.00859923329618242, 0.013474992745452457, 0.02209889358944363, 0.007023960517512427]
        write_times = [0.006543803049458397, 0.00801571508248647, 0.013281303975317214, 0.01996548722187678, 0.03248203727934096, 0.06818085643980239, 0.12449431485599942]
        q3s = [0, 0.0037678084607014606, 0.012110132701495857, 0.019185196767860374, 0.027698495848216792, 0.049346622701334886, 0.13205600333701964]
        p95s = [0.012897164696479932, 0.023066979115367023, 0.03746925253138035, 0.049276048124138716, 0.0633938838821543, 0.10607286090415023, 0.2682467369963562]
        p99s = [0.029310025826137665, 0.044204964098611944, 0.06363057617679498, 0.07863426104669242, 0.09764280944131998, 0.19566193556584022, 0.37161204950799337]
        vars = [ 0.000035201866799052135, 0.00008734450324067557, 0.00019680877880315966, 0.0003172759904348628, 0.0005133524415211174, 0.001622537064078928, 0.0072993933328457175]
        stdevs = [0.005933116112048721, 0.009345828119576969, 0.0140288552206928, 0.0178122427121029, 0.022657282306603268, 0.04028072819697936, 0.0854364871284261]

        # chart
        reset_plt_font_scale()
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # ref
        color = 'darkorange'
        plt.hlines(y=18804960, xmin=0, xmax=100, color=color, linestyles=':', linewidth=1)
        plt.text(95, 18904960, "Raw storage", c=color, fontsize=7)

        color = 'tab:red'
        ax1.set_xlabel('Count of series per submodel')
        ax1.set_ylabel('Complete model size (# floats)', color=color)
        ax1.plot(series_packet_size, sizes, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.plot(series_packet_size, p95s, color=color)
        ax2.set_ylabel('P95 value', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        color = 'tab:green'
        ax3 = ax1.twinx()
        ax3.plot(series_packet_size, p99s, color=color)
        ax3.set_ylabel('P99 value', color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.spines['right'].set_position(('outward', 60))

        fig.tight_layout()
        gca = plt.gca()
        gca.invert_xaxis()
        return gca

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare techniques with the same modelling error
    """)
    return


@app.cell(hide_code=True)
def _(SplitFLInD, electricity_df, electricity_series):
    # Big benchmark (not run by default since it takes so much time)
    def _():
        return
        f = electricity_df
        c = electricity_series
        dimensions_count = len(c)

        f = normalize(f)

        iterations_count = 1
        names = []
        errors = []
        sizes = []
        p95s = []
        read_times = []
        write_times = []
        compression_times = []

        for i in range(iterations_count):
            print(f'Iteration #{i}')

            # hardware target: MSE=0.001
            # FLInDwFLI: 0.695
            # FLInD: 2.013
            # SplitFLInD: 1.8

            # alibaba target: MSE=0.001
            # FLInDwFLI: 0.4
            # FLInD: 2.75
            # SplitFLInD: 0.98

            # solar target: MSE=0.001
            # FLInDwFLI: 0.25
            # FLInD: 0.864
            # SplitFLInD: 0.64

            # electricity
            # FLInDwFLI: 0.43
            # FLInD: 1.802
            # SplitFLInD: 1.05


            # hardware
            # model_errors = [0.0,
            #                 0.0, 0.232, 0.464, 0.695, 0.928,
            #                 0.0, 0.671, 1.342, 2.013, 2.684,
            #                 0.0, 0.6, 1.2, 1.8, 2.4]

            # alibaba
            # model_errors = [0.0,
            #                 0.0, 0.13, 0.26, 0.4, 0.53,
            #                 0.0, 0.92, 1.84, 2.75, 3.68,
            #                 0.0, 0.33, 0.66, 0.98, 1.33]

            # solar power
            # model_errors = [0.0,
            #                 0.0, 0.8, 0.16, 0.25, 0.34,
            #                 0.0, 0.288, 0.576, 0.864, 1.152,
            #                 0.0, 0.21, 0.42, 0.64, 0.85]

            # electricity
            model_errors = [#0.0,
                            #0.0, 0.14, 0.28, 0.43, 0.57,]
                            #0.0, 0.6, 1.2, 1.802, 2.4,]
                            0.0, 0.35, 0.7, 1.05, 1.4]

            models = [
                #RawStorage(dimensions_count),
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[1])),
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[2])),
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[3])),
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[4])),
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[5])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[0])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[1])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[2])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[3])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[4])),
                SplitFLInD(dimensions_count, 50, model_errors[0]),
                SplitFLInD(dimensions_count, 50, model_errors[1]),
                SplitFLInD(dimensions_count, 50, model_errors[2]),
                SplitFLInD(dimensions_count, 50, model_errors[3]),
                SplitFLInD(dimensions_count, 50, model_errors[4]),

                # # lossless
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[1])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[2])),
                # SplitFLInD(dimensions_count, 50, model_errors[3]),

                # # lossy
                # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[4])),
                # FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[5])),
                # SplitFLInD(dimensions_count, 50, model_errors[6]),
                # CliqueFLInD(dimensions_count, 200, 0.1, computeStdErrorsArray(f, model_errors[3])),
                # TimescaleDB(alibaba_series) # max column count = 1600, not enough to fit alibaba
            ]
            assert len(models) == len(model_errors)
            frame = bench_dimensional_models(models, f, c, model_errors, True)[0]
            read_times.append(np.array(frame["Read time"]))
            write_times.append(np.array(frame["Write time"]))
            compression_times.append(np.array(frame["Compression time"]))

            if i == 0:
                names = frame['Name']
                errors = [f'x{x}' for x in frame['Error']]
                sizes = frame['Size']
                p95s = frame['P95']
                mse = frame['MSE']

        read_averages = [round(1/np.mean(x)) for x in zip(*read_times)]
        write_averages = [round(1/np.mean(x)) for x in zip(*write_times)]
        compression_averages = [np.mean(x) for x in zip(*compression_times)]
        #read_averages = ["{0:.3e}".format(x) for x in read_averages]
        #write_averages = ["{0:.3e}".format(x) for x in write_averages]
        compression_averages = ["{0:.3e}".format(x) for x in compression_averages]

        # Convert sizes
        sizes = [human_readable_size(s) for s in sizes]

        return pl.DataFrame({
            'Name': names,
            'Error': errors,
            'Size': sizes,
            'Reads/sec': read_averages,
            'Write/sec': write_averages,
            'P95': p95s,
            'MSE': mse,
            'Compression time': compression_averages
        })

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model size evolution with error progression
    """)
    return


@app.cell(hide_code=True)
def _(alibaba_df, alibaba_series, computeStdErrorsArray):
    # Big benchmark (not run by default since it takes so much time)
    def _():
        #return
        f = alibaba_df
        c = alibaba_series
        dimensions_count = len(c)

        f = normalize(f)

        sizes = []
        p95s = []
        read_times = []
        write_times = []
        compression_times = []

        # hardware target: MSE=0.001
        # FLInDwFLI: 0.695
        # FLInD: 2.013
        # SplitFLInD: 1.8

        # alibaba target: MSE=0.001
        # FLInDwFLI: 0.4
        # FLInD: 2.75
        # SplitFLInD: 0.98

        # solar target: MSE=0.001
        # FLInDwFLI: 0.25
        # FLInD: 0.864
        # SplitFLInD: 0.64

        # electricity
        # FLInDwFLI: 0.43
        # FLInD: 1.802
        # SplitFLInD: 1.05

        hardware_model_errors = [0.0, 0.695, 1.39, 2.78, 5.56, 2.013, 4.026, 8.052, 16.104, 1.8, 3.6, 7.2, 14.4]
        solar_model_errors = [0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 0.864, 1.728, 3.456, 6.912, 13.824, 0.64, 1.28, 2.56, 5.12, 10.24]
        elec_model_errors = [0.0, 0.43, 0.86, 1.72, 3.44, 1.802, 3.604, 7.208, 14.416, 1.05, 2.1, 4.2, 8.4]
        alibaba_model_errors = [0.0, 0.4, 0.8, 1.6, 3.2, 2.75, 5.5, 11, 22, 0.98, 1.96, 3.92, 7.84]
        model_errors = [0.92, 1.84, 3.68]

        models = [
            #RawStorage(dimensions_count),

            # fli
            # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[1])),
            # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[2])),
            # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[3])),
            # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[4])),
            # FLInDwFLI(dimensions_count, computeStdErrorsArray(f, model_errors[5])),

            # flind
            FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[0])),
            FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[1])),
            FLInD(dimensions_count, computeStdErrorsArray(f, model_errors[2])),

            # split
            # SplitFLInD(dimensions_count, 50, model_errors[11]),
            # SplitFLInD(dimensions_count, 50, model_errors[12]),
            # SplitFLInD(dimensions_count, 50, model_errors[13]),
            # SplitFLInD(dimensions_count, 50, model_errors[14]),
            # SplitFLInD(dimensions_count, 50, model_errors[15]),
            # CliqueFLInD(dimensions_count, 200, 0.1, computeStdErrorsArray(f, model_errors[3])),
            # TimescaleDB(alibaba_series) # max column count = 1600, not enough to fit alibaba
        ]

        assert len(models) == len(model_errors)
        frame = bench_dimensional_models(models, f, c, model_errors, True)[0]
        return frame

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Recap figures
    """)
    return


@app.cell(hide_code=True)
def _():
    # Recap clustered sizes
    def _():
        datasets = ("Hardware profile", "Electricity load", "Solar power", "Microservices")

        # Raw measurements
        technique_sizes = {
            'Raw': (39.40, 397.00, 325.6, 17.93),
            'TimescaleDB': (43.05, 548.05, 410.7, 0),
            'FLI': (44.08, 614.3, 268.03, 35.83),
            'FLInD (ours)': (39.39, 396.99, 169.86, 17.91),
            'Clique (ours)': (25.16, 396.84, 169.86, 17.91),
            'Split (ours) (P95 < 0.05)': (17.59, 156.7, 0, 6.33)
        }

        # Center chart using raw storage value as pivot
        for technique, sizes in technique_sizes.items():
            if technique == 'Raw':
                continue
            new_values = []
            for index, value in enumerate(sizes):
                new_values.append(value / technique_sizes['Raw'][index])
            technique_sizes[technique] = new_values
        technique_sizes['Raw'] = tuple([1 for x in range(len(datasets))])

        l = len(technique_sizes.keys())
        x = np.arange(len(datasets))  # the label locations
        width = 1 / (l + 1)  # the width of the bars
        multiplier = 0

        # Font sizes
        plt.rcParams.update({'font.size': 8 if l < 5 else 6})
        plt.rc('axes', titlesize=10) # title
        plt.rc('axes', labelsize=8)  # y-axis
        plt.rc('xtick', labelsize=8) # x-axis
        plt.rc('legend', fontsize=7)
        fig, ax = plt.subplots(layout='constrained', figsize=(8, 2), dpi=200)

        # Do not display size ratios with full precision
        def format_size_ratio(r) -> str:
            if r == 0:
                return '$\emptyset$'
            if r == 1:
                return 'x1'

            formatted = f'x{"{:.2f}".format(r)}'
            if formatted[3:] == '00':
                return formatted[:2]
            return formatted

        for attribute, measurement in technique_sizes.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3, labels=[format_size_ratio(m) for m in measurement])
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.set_ylabel('Storage cost (MiB)')
        ax.set_ylabel('Compression ratio')
        ax.set_title('Dataset size comparison per technique (lower is better)')
        ax.set_xticks(x + width*(l-1)/2, datasets)
        ax.legend(loc='upper left', ncols=l)
        ax.set_ylim(0, 2.25)

        plt.savefig('size_comparison.pdf')
        return plt.gca()

    _()
    return


@app.cell(hide_code=True)
def _():
    # Compare error integration in FLI and FLInD
    def _():
        # data
        std_counts = [1, 2, 5, 10, 20]
        fli_sizes = [182.63, 128.99, 63.77, 28.99, 10.39]
        flind_sizes = [143.27, 143.27, 143.27, 143.27, 143.27]

        # plot
        ## Font sizes
        plt.rcParams.update({'font.size': 8})
        plt.rc('axes', titlesize=10) # title
        plt.rc('axes', labelsize=7)
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=6)
        plt.rc('legend', fontsize=6)
        fig, ax = plt.subplots(figsize=(3, 1.5), dpi=200)

        ax.plot(std_counts, fli_sizes, label="Separate (FLI)")
        ax.plot(std_counts, flind_sizes, label="Combined (FLInD)")
        ax.set_ylabel('Model size (MiB)')
        ax.set_xlabel('Tolerated error (in counts of standard deviations per series)')
        ax.set_xticks(std_counts, [f'x{c}' for c in std_counts])
        ax.set_title('Error integration in linear regression models')
        ax.legend()
        plt.savefig('error_in_fli_vs_flind.pdf', bbox_inches='tight')

        return plt.gca()

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Error integration
    """)
    return


@app.cell(hide_code=True)
def _():
    def _():
        # data
        hardware_fli_errors = [0.695, 1.39, 2.78, 5.56]
        hardware_flind_errors = [2.013, 4.026, 8.052, 16.104]
        hardware_split_errors = [1.8, 3.6, 7.2, 14.4]
        hardware_fli_sizes = [16214112, 10049776, 1925680, 253488]
        hardware_flind_sizes = [37443888, 10672832, 1678752, 633016]
        hardware_split_sizes = [29832960, 9893592, 2263448, 478312]
        hardware_fli_mse = [0.0012106019365737148, 0.009832734272506236, 0.045128106915738646, 0.06356706274834695]
        hardware_flind_mse = [0.001386622799770866, 0.021827358324070685, 0.02992133492818222, 0.0318265928150929]
        hardware_split_mse = [0.0010604095506424628, 0.02107568555003403, 0.03211654502342431, 0.034993629030658745]

        electricity_fli_errors = [0.43, 0.86, 1.72, 3.44]
        electricity_flind_errors = [1.802, 3.604, 7.208, 14.416]
        electricity_split_errors = [1.05, 2.1, 4.2, 8.4]
        electricity_fli_sizes = [78546032, 30525952, 10543232, 846480]
        electricity_flind_sizes = [105417424, 29849176, 2819600, 448168]
        electricity_split_sizes = [93771888, 24349560, 4177104, 299640]
        electricity_fli_mse = [0.001039956075945359, 0.004891172688246059, 0.02301026985731188, 0.0705099149603604]
        electricity_flind_mse = [0.0010417245946885059, 0.008299291719285461, 0.026317251829191854, 0.03184178323771506]
        electricity_split_mse = [0.0010250327680900654, 0.006615461436439918, 0.026176340820260782, 0.04656048948496262]

        # solar_fli_errors = [0.25, 0.5, 1.0, 2.0]
        # solar_flind_errors = [0.864, 1.728, 3.456, 6.912]
        # solar_split_errors = [0.64, 1.28, 2.56, 5.12]
        # solar_fli_sizes = [26701856, 15916800, 8496400, 2648544]
        # solar_flind_sizes = [57236256, 12547024, 555408, 3248]
        # solar_split_sizes = [40939080, 13032600, 1759536, 3312]
        # solar_fli_mse = [0.0011871748100890146, 0.006746023430958831, 0.02938920678946968, 0.17995224885274885]
        # solar_flind_mse = [0.0010068896176588713, 0.01943612862046303, 0.18291318366543563, 0.12703348333237166]
        # solar_split_mse = [0.0010025613843316328, 0.009761945222219223, 0.1604543072684912, 0.12703348333237166]
        solar_fli_errors = [0.25, 0.5, 1.0, 2.0, 4.0]
        solar_flind_errors = [0.864, 1.728, 3.456, 6.912, 13.824]
        solar_split_errors = [0.64, 1.28, 2.56, 5.12, 10.24]
        solar_fli_sizes = [26701856, 15916800, 8496400, 2648544, 6528]
        solar_flind_sizes = [57236256, 12547024, 555408, 3248, 3248]
        solar_split_sizes = [40939080, 13032600, 1759536, 3312, 3312]
        solar_fli_mse = [0.0011871748100890146, 0.006746023430958831, 0.02938920678946968, 0.17995224885274885, 0.1277207640846566]
        solar_flind_mse = [0.0010068896176588713, 0.01943612862046303, 0.18291318366543563, 0.12703348333237166, 0.12703348333237166]
        solar_split_mse = [0.0010025613843316328, 0.009761945222219223, 0.1604543072684912, 0.12703348333237166, 0.12703348333237166]

        alibaba_fli_errors = [0.695, 1.39, 2.78, 5.56]
        alibaba_fli_sizes = [32574032, 11287072, 2470512, 342928]
        alibaba_flind_errors = [2.013, 4.026, 8.052, 16.104]
        alibaba_flind_sizes = [116381808, 39281472, 6268320, 313416]
        alibaba_split_errors = [1.8, 3.6, 7.2, 14.4]
        alibaba_split_sizes = [26318112, 5316312, 299136, 107376]
        alibaba_fli_mse = [0.0031764539299991013, 0.01156980851484502, 0.039582510854624366, 0.06398005561739514]
        alibaba_flind_mse = [0.0002351881571429652, 0.0027566018820463323, 0.013733715434807676, 0.033395481365707115]
        alibaba_split_mse = [0.0038976910093267082, 0.01373395553401054, 0.046035659862229905, 0.05633994629319084]


        # plot
        errors = [1, 2, 4, 8]
        plt.figure(layout='constrained', figsize=(10, 4), dpi=200)
        xlabel = 'Counts of <MSE=0.001> errors'

        ## hardware
        plt.subplot(241)
        plt.title('Hardware')
        # plt.plot(errors, hardware_fli_sizes, label='FLI')
        # plt.plot(errors, hardware_flind_sizes, label='FLInD')
        # plt.plot(errors, hardware_split_sizes, label='Split')
        # plt.xticks(errors, [f'x{e}' for e in errors])
        # plt.xlabel(xlabel)
        # plt.ylabel('Storage budget (bytes)')
        plt.plot(hardware_fli_mse, hardware_fli_sizes, label='FLI')
        plt.plot(hardware_flind_mse, hardware_flind_sizes, label='FLInD')
        plt.plot(hardware_split_mse, hardware_split_sizes, label='Split')
        plt.xlabel('Observed MSE')
        plt.ylabel('Storage budget (bytes)')
        plt.yscale('log')
        plt.legend()

        plt.subplot(245)
        plt.plot(errors, hardware_fli_mse, label='FLI')
        plt.plot(errors, hardware_flind_mse, label='FLInD')
        plt.plot(errors, hardware_split_mse, label='Split')
        plt.xticks(errors, [f'x{e}' for e in errors])
        plt.xlabel(xlabel)
        plt.ylabel('Observed MSE')
        plt.legend()

        ## electricity
        plt.subplot(242)
        plt.title('Electricity')
        # plt.plot(errors, electricity_fli_sizes, label='FLI')
        # plt.plot(errors, electricity_flind_sizes, label='FLInD')
        # plt.plot(errors, electricity_split_sizes, label='Split')
        # plt.xticks(errors, [f'x{e}' for e in errors])
        # plt.xlabel(xlabel)
        plt.plot(electricity_fli_mse, electricity_fli_sizes, label='FLI')
        plt.plot(electricity_flind_mse, electricity_flind_sizes, label='FLInD')
        plt.plot(electricity_split_mse, electricity_split_sizes, label='Split')
        plt.xlabel('Observed MSE')
        plt.yscale('log')
        plt.legend()

        plt.subplot(246)
        plt.plot(errors, electricity_fli_mse, label='FLI')
        plt.plot(errors, electricity_flind_mse, label='FLInD')
        plt.plot(errors, electricity_split_mse, label='Split')
        plt.xticks(errors, [f'x{e}' for e in errors])
        plt.xlabel(xlabel)
        plt.legend()

        ## solar
        solar_errors = [1, 2, 4, 8, 16]
        plt.subplot(243)
        plt.title('Solar power')
        # plt.plot(solar_errors, solar_fli_sizes, label='FLI')
        # plt.plot(solar_errors, solar_flind_sizes, label='FLInD')
        # plt.plot(solar_errors, solar_split_sizes, label='Split')
        # plt.xticks(solar_errors, [f'x{e}' for e in solar_errors])
        # plt.xlabel(xlabel)
        plt.plot(solar_fli_mse, solar_fli_sizes, label='FLI')
        plt.plot(solar_flind_mse, solar_flind_sizes, label='FLInD')
        plt.plot(solar_split_mse, solar_split_sizes, label='Split')
        plt.xlabel('Observed MSE')
        plt.yscale('log')
        plt.legend()

        plt.subplot(247)
        plt.plot(solar_errors, solar_fli_mse, label='FLI')
        plt.plot(solar_errors, solar_flind_mse, label='FLInD')
        plt.plot(solar_errors, solar_split_mse, label='Split')
        plt.xticks(solar_errors, [f'x{e}' for e in solar_errors])
        plt.xlabel(xlabel)
        plt.legend()

        ## microservices
        plt.subplot(244)
        plt.title('Microservices')
        # plt.plot(errors, alibaba_fli_sizes, label='FLI')
        # plt.plot(errors, alibaba_flind_sizes, label='FLInD')
        # plt.plot(errors, alibaba_split_sizes, label='Split')
        # plt.xticks(errors, [f'x{e}' for e in errors])
        # plt.xlabel(xlabel)
        plt.plot(alibaba_fli_mse, alibaba_fli_sizes, label='FLI')
        plt.plot(alibaba_flind_mse, alibaba_flind_sizes, label='FLInD')
        plt.plot(alibaba_split_mse, alibaba_split_sizes, label='Split')
        plt.xlabel('Observed MSE')
        plt.yscale('log')
        plt.legend()

        plt.subplot(248)
        plt.plot(errors, alibaba_fli_mse, label='FLI')
        plt.plot(errors, alibaba_flind_mse, label='FLInD')
        plt.plot(errors, alibaba_split_mse, label='Split')
        plt.xticks(errors, [f'x{e}' for e in errors])
        plt.xlabel(xlabel)
        plt.legend()

        plt.savefig('mse_comparison.pdf')
        return plt.gca()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## I/O comparison recap
    """)
    return


@app.cell(hide_code=True)
def _():
    def _():
        # data
        models = ['Raw', 'FLI', 'FLInD', 'Split', 'FLI (lossy)', 'FLInD (lossy)', 'Split (lossy)']
        colors = ['nope', 'C0', 'C1', 'C2', 'C0', 'C1', 'C2']
        ## hardware
        hardware_sizes = [39.40, 44.35, 39.39, 39.93, 15.46, 35.71, 28.45]
        hardware_reads = [1447618, 7802, 50272, 36325, 9481, 50369, 39829]
        hardware_writes = [106612, 27772, 33220, 20288, 26443, 32594, 20542]
        ## electricity
        electricity_sizes = [397.00, 615.64, 396.99, 371.53, 74.91, 100.53, 89.43]
        electricity_reads = [722594, 808, 14132, 8214, 1004, 14013, 8578]
        electricity_writes = [42892, 5205, 13201, 4892, 5486, 13777, 5298]
        # solar
        solar_sizes = [325.61, 271.83, 169.90, 166.72, 25.46, 54.58, 39.04]
        solar_reads = [616524, 682, 12589, 7303, 932, 12512, 7613]
        solar_writes = [39182, 5363, 12673, 4343, 5098, 12783, 4353]
        # microservices
        microservices_sizes = [143.47, 286.65, 143.27, 146.13, 58.75, 66.85, 61.23]
        microservices_reads = [15687, 41, 424, 237, 37, 420, 250]
        microservices_writes = [683, 118, 344, 126, 136, 351, 129]

        # plot
        plt.figure(layout='constrained', figsize=(10, 3), dpi=200)
        ## hardware
        plt.subplot(241)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(hardware_sizes[i], hardware_reads[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(hardware_sizes[0], ymin=min(hardware_reads[1:]), ymax=max(hardware_reads[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.title('Hardware')
        plt.ylabel('Reads per second')
        plt.legend()
        # --
        plt.subplot(245)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(hardware_sizes[i], hardware_writes[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
            plt.vlines(hardware_sizes[0], ymin=min(hardware_writes[1:]), ymax=max(hardware_writes[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.ylabel('Writes per second')
        plt.xlabel('Storage budget (MiB)')

        ## electricity
        plt.subplot(242)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(electricity_sizes[i], electricity_reads[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(electricity_sizes[0], ymin=min(electricity_reads[1:]), ymax=max(electricity_reads[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.title('Electricity')
        # --
        plt.subplot(246)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(electricity_sizes[i], electricity_writes[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(electricity_sizes[0], ymin=min(electricity_writes[1:]), ymax=max(electricity_writes[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.xlabel('Storage budget (MiB)')

        ## solar
        plt.subplot(243)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(solar_sizes[i], solar_reads[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(solar_sizes[0], ymin=min(solar_reads[1:]), ymax=max(solar_reads[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.title('Solar power')
        # --
        plt.subplot(247)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(solar_sizes[i], solar_writes[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(solar_sizes[0], ymin=min(solar_writes[1:]), ymax=max(solar_writes[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.xlabel('Storage budget (MiB)')

        ## microservices
        plt.subplot(244)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(microservices_sizes[i], microservices_reads[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(microservices_sizes[0], ymin=min(microservices_reads[1:]), ymax=max(microservices_reads[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.title('Microservices')
        # --
        plt.subplot(248)
        for i,v in enumerate(models):
            if i == 0:
                continue
            is_lossy = True if i >= 4 else False
            plt.scatter(microservices_sizes[i], microservices_writes[i], label=models[i], marker="o", hatch='//////' if is_lossy else '', color=colors[i], facecolor='white' if is_lossy else colors[i])
        plt.vlines(microservices_sizes[0], ymin=min(microservices_writes[1:]), ymax=max(microservices_writes[1:]), color='red', linestyles='dotted', linewidth=1)
        plt.xlabel('Storage budget (MiB)')

        plt.savefig('comparison_recap.pdf')
        return plt.gca()

    _()
    return


@app.cell(hide_code=True)
def _():
    # Same, but generated from CSV files
    def _():
        results = [
            { 'name': 'Hardware', 'path': '~/Nextcloud/Inria/FLAIR/Tables/Size recap/final/hardware all 4iter.csv' },
            { 'name': 'Electricity', 'path': '~/Nextcloud/Inria/FLAIR/Tables/Size recap/final/elec compil 1iter.csv' },
            { 'name': 'Solar power', 'path': '~/Nextcloud/Inria/FLAIR/Tables/Size recap/final/solar all 4iter 93min.csv' },
            { 'name': 'Microservices', 'path': '~/Nextcloud/Inria/FLAIR/Tables/Size recap/final/alibaba all 4iter.csv' }
        ]
        plot_args = {
            'marker': '.',
            'markersize': 4,
            'linewidth': 1
        }
        display_error_progress_lines = True

        def get_algo_label(alg):
            if 'FLI (' in alg:
                return 'FLI'
            if 'Split' in alg:
                return 'Split'
            return 'FLInD'

        def get_algo_color(alg):
            label = get_algo_label(alg)
            if label == 'FLI':
                return 'C0'
            if label == 'Split':
                return 'C2'
            return 'C1'

        plt.figure(layout='constrained', figsize=(10, 3), dpi=200)
        for index, result in enumerate(results):
            frame = pl.read_csv(result['path'])
            algorithms = frame['Name'].unique(maintain_order=True)
            baseline_storage_budget = 0
            max_reads = 0
            min_reads = 0
            max_writes = 0
            min_writes = 0

            for algo_index, algorithm in enumerate(algorithms):
                algo_results = frame.filter(pl.col('Name') == algorithm)
                alg_color = get_algo_color(algorithm)

                if algorithm == "Raw storage":
                    baseline_storage_budget = float(algo_results['Size'][0][:-4]) # remove "MiB suffix"
                    max_reads = frame.filter(pl.col('Name') != algorithm)['Reads/sec'].max()
                    min_reads = frame.filter(pl.col('Name') != algorithm)['Reads/sec'].min()
                    max_writes = frame.filter(pl.col('Name') != algorithm)['Write/sec'].max()
                    min_writes = frame.filter(pl.col('Name') != algorithm)['Write/sec'].min()
                    continue

                # reads/sec
                plt.subplot(int(f'24{index+1}'))
                plt.title(result['name'])
                algo_sizes = [float(s[:-4]) for s in algo_results['Size']] # remove "MiB suffix"
                algo_reads = algo_results['Reads/sec']
                if display_error_progress_lines:
                    plt.plot(algo_sizes, algo_reads, label=get_algo_label(algorithm), color=alg_color, **plot_args)

                ## Lossless value
                plt.scatter(algo_sizes[0], algo_reads[0], marker="o", color=alg_color, label=get_algo_label(algorithm) if not display_error_progress_lines else '')
                ## Comparable lossy value (with MSE ~= 0.001) SHOULD BE THE THIRD VALUE FOR EACH ALGORITHM!
                plt.scatter(algo_sizes[3], algo_reads[3], marker="o", hatch='//////', facecolor='white', color=alg_color)

                ## only display legend on top-left chart
                if index == 0:
                    plt.legend()
                    if algo_index == 1:
                        plt.ylabel('Reads per second')

                # writes/sec
                plt.subplot(int(f'24{index+1+4}'))
                algo_writes = algo_results['Write/sec']
                if display_error_progress_lines:
                    plt.plot(algo_sizes, algo_writes, color=alg_color, **plot_args)
                plt.xlabel('Storage budget (MiB)')

                ## Lossless value
                plt.scatter(algo_sizes[0], algo_writes[0], marker="o", color=alg_color)
                ## Comparable lossy value (with MSE ~= 0.001) SHOULD BE THE THIRD VALUE FOR EACH ALGORITHM!
                plt.scatter(algo_sizes[3], algo_writes[3], marker="o", hatch='//////', facecolor='white', color=alg_color)

                ## only display y-label on most leftern charts
                if index == 0 and algo_index == 1:
                    plt.ylabel('Writes per second')

            # baseline storage budget
            plt.subplot(int(f'24{index+1}'))
            plt.vlines(baseline_storage_budget, ymin=min_reads, ymax=max_reads, color='red', linestyles='dotted', linewidth=1)
            plt.subplot(int(f'24{index+1+4}'))
            plt.vlines(baseline_storage_budget, ymin=min_writes, ymax=max_writes, color='red', linestyles='dotted', linewidth=1)

        plt.savefig('comparison_recap.pdf')
        return plt.gca()

    _()
    return


if __name__ == "__main__":
    app.run()
