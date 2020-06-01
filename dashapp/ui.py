import base64
import io
import json
import os
import pickle
import numpy as np
import dash_table
from natsort import natsorted
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from pathlib import Path

from mskit.spectrum import Spectrum
from mskit.MS_Classification_CSV import calc_area_stdev, make_meshgrid, plot_contours, plot_svm, cluster_intensities, ward_clustering, predict_X, plot_clustergram

def _read_upload(upload, filename):
    _, extension = os.path.splitext(filename)
    _, content_bitstring = upload.split(",")
    content = base64.b64decode(content_bitstring)
    # content is now a bytestring

    if extension == ".csv":
        content = content.decode("utf-8")  # bytestring -> string
        signal = np.loadtxt(io.StringIO(content), delimiter = ',')
        spectrum = Spectrum(signal=signal)
        return spectrum
    else:
        raise ValueError(f"Unknown format: {extension}")


def parse_upload(contents_list, filename_list):
    """Convert uploaded files to objects

    """
    spectrum_data_list = []

    if contents_list:
        idx = 0

        # Single spectrum uploaded
        if type(filename_list) != list:
            spectrum = _read_upload(contents_list, filename_list)
            spectrum.idx = idx
            spectrum.label = filename_list.split(".")[0]
            spectrum_data_list.append(spectrum.to_json())

        else:
            for contents, filename in zip(contents_list, filename_list):
                spectrum = _read_upload(contents, filename)
                spectrum.idx = idx
                spectrum.label = filename.split(".")[0]
                spectrum_data_list.append(spectrum.to_json())
                idx += 1

    return spectrum_data_list


def get_spectrum_plot(spectrum_json_list):
    if not spectrum_json_list:
        return html.Div()

    spectrum_list = [
        Spectrum.from_json(spectrum_json) for spectrum_json in spectrum_json_list
    ]

    traces = []
    for i, spectrum in enumerate(natsorted(spectrum_list, key = lambda spectrum: spectrum.label)):
        spectrum.idx = i + 1
        skip = len(spectrum.s) / 5000
        traces.append(
            go.Scatter(
                x=spectrum.t[::int(skip)],
                y=spectrum.s[::int(skip)],
                name=f"Spectrum {spectrum.idx:02d}: {spectrum.label}",
                mode = 'lines'
            )
        )

    layout = go.Layout(
        xaxis={"title": "t"}, yaxis={"title": "Total Intensity"}, hovermode="closest"
    )

    return html.Div([dcc.Graph(id="plot1", figure={"data": traces, "layout": layout})])

def get_svm_plot(spectrum_json_list):
    if not spectrum_json_list:
        return html.Div()

    spectrum_list = [
        Spectrum.from_json(spectrum_json) for spectrum_json in spectrum_json_list
    ]
    a_sd = []
    spectrum_list = natsorted(spectrum_list, key = lambda spectrum: spectrum.label)

    a_sd = [calc_area_stdev(spectrum.s) for spectrum in spectrum_list]
    areas = [i[0] for i in a_sd]
    stdevs = [i[1] for i in a_sd]
    p = Path(__file__).parent.parent / 'svm_clf_unstd'
    p = p.absolute()

    clf = pickle.load(open(p, 'rb'))
    traces, predictions = (plot_svm(clf, [(str(i + 1) + ": " + spectrum.label) for i, spectrum in enumerate(spectrum_list)]))

    good = [(str(i + 1) + ": " + spectrum_list[i].label) for i, val in enumerate(predictions) if val == 1]
    bad = [(str(i + 1) + ": " + spectrum_list[i].label) for i, val in enumerate(predictions) if val == 0]

    layout = go.Layout(
        xaxis={"title": "Scaled Area Under Curve"}, yaxis={"title": "Scaled Standard Deviation"}, hovermode="closest"
    )

    return html.Div([dcc.Graph(id="plot2", figure={"data": traces, "layout": layout}), html.P('Good: ' + str(good)), html.P('Bad: ' + str(bad))])

def get_full_ward_plot(spectrum_json_list):
    if not spectrum_json_list:
        return html.Div()

    spectrum_list = [
        Spectrum.from_json(spectrum_json) for spectrum_json in spectrum_json_list
    ]
    spectrum_list = natsorted(spectrum_list, key = lambda spectrum: spectrum.label)

    traces = []
    full_clustered = []
    for spectrum in spectrum_list:
        time = np.array(spectrum.t)
        inten = np.array(spectrum.s)
        temp_time_inten = np.vstack((time, inten))
        temp_time_inten = temp_time_inten.transpose()
        full_clustered.append(cluster_intensities(temp_time_inten))
    traces.append(ward_clustering(full_clustered, [(str(i + 1) + ": " + spectrum.label) for i, spectrum in enumerate(spectrum_list)]))

    layout = go.Layout(
        xaxis={"title": "Scaled Area Under Curve"}, yaxis={"title": "Scaled Standard Deviation"}, hovermode="closest"
    )

    return html.Div([dcc.Graph(id="plot3", figure={"data": traces, "layout": layout})])

def get_dendrogram_plot(spectrum_json_list):
    if not spectrum_json_list:
        return html.Div()

    spectrum_list = [
        Spectrum.from_json(spectrum_json) for spectrum_json in spectrum_json_list
    ]
    spectrum_list = natsorted(spectrum_list, key = lambda spectrum: spectrum.label)

    full_clustered = []
    for spectrum in spectrum_list:
        time = np.array(spectrum.t)
        inten = np.array(spectrum.s)
        temp_time_inten = np.vstack((time, inten))
        temp_time_inten = temp_time_inten.transpose()
        full_clustered.append(cluster_intensities(temp_time_inten))
    full_clustered = np.nan_to_num(full_clustered, copy = False)
    clustergram, cluster_table = plot_clustergram(full_clustered, len(full_clustered), 'average')
    print(cluster_table)

    layout = go.Layout(
        hovermode="closest"
    )

    return html.Div([dcc.Graph(id = "plot5", figure = clustergram), 
        html.P(str(cluster_table))])