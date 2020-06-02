import wrangling
import prediction
import training
import constants
import os
from urllib.parse import quote as urlquote
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
from flask import Flask, send_from_directory
from random import random
import plotly.express as px
import numpy as np

os.chdir('F:/5500_P1_MVP/data')
default_train_df = pd.read_csv('default_train_df.csv')
default_feature_importance = pd.read_csv('default_feature_importance.csv')
default_figure_feature_importance = {
            'data': [
                {'x': default_feature_importance['m/z scope'].tolist(), 'y': default_feature_importance['importance'].tolist(), 'type': 'bar'}
            ],
            'layout': {
                'title': 'feature importance will be displayed below'
            }
        }
default_roc_curve = px.line(default_train_df, x="fpr", y="tpr", title='ROC curve will be displayed below')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Mass spec spectrum QC"),
    html.H2("File directory input"),
    # html.Div(dcc.Input(id='input-on-submit', type='text', value='F:/5500_P1_MVP/')), #add an input bar
    html.Div(dcc.Input(id='input-on-submit', type='text',
                       value='input your directory here')),  # add an input bar
    html.H6("---------------------------------------------------------------------------------------------"),
    html.H2("Training"),
    html.H4('Hyper-parameter tuning scope selection'),
    html.H6("[NUMBER OF ESTIMATORS]"),
    html.Div(id='output-container-range-slider'),
    dcc.RangeSlider(
        id='n_estimators_RangeSlider',
        min=50,
        max=250,
        step=50,
        value=[50, 100],marks = {50:'50',100:'100',150:'150',200:'200',250:'250'}
    ),
    html.H6("[MINIMAL NUMBER OF SAMPLES IN A LEAF]"),
    html.Div(id='output-min_samples_leaf_RangeSlider'),
    dcc.RangeSlider(
        id='min_samples_leaf_RangeSlider',
        min=2,
        max=32,
        step=2,
        value=[2, 8],marks = {2:'2',4:'4',8:'8',16:'16',32:'32'}
    ),
    html.Button('Start/stop training', id='submit-val', n_clicks=0),  # add a button to start training
    daq.Indicator(id='train-indicator',label="Training in process",value=True,color='grey'),
    html.Div(id='spectrum quality training',children='AUC score:___'),  # add a section to store and display output
    html.H6("---------------------------------------------------------------------------------------------"),
    html.H2("Prediction"),
    html.Button('Start/stop prediction', id='submit-pred', n_clicks=0),  # add a button to start training
    daq.Indicator(id='predict-indicator',label="Prediction in process",value=True,color='grey'),
    html.Div(id='spectrum quality prediction',children='prediction result-positive rate:___'),  # add a section to store and display output
    # html.H4("File List"),
    # html.Ul(id="file-list"),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value-1', style={'display': 'none'}),
    html.Div(id='intermediate-value-2', style={'display': 'none'}),
    html.H6("---------------------------------------------------------------------------------------------"),
    html.H2("Visualization"),
    html.H4("Feature importance"),
    dcc.Graph(id='important-features-graph',figure=default_figure_feature_importance),
    html.H4("ROC curve plot"),
    dcc.Graph(id='ROC curve',figure = default_roc_curve)

])


@app.callback(
    [dash.dependencies.Output('important-features-graph', 'figure'),
     dash.dependencies.Output('spectrum quality training', 'children'),
     dash.dependencies.Output('ROC curve','figure'),
     dash.dependencies.Output('output-container-range-slider','children'),
     dash.dependencies.Output('output-min_samples_leaf_RangeSlider','children')],
    # specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-val', 'n_clicks'),
     dash.dependencies.Input('n_estimators_RangeSlider', 'value'),
     dash.dependencies.Input('min_samples_leaf_RangeSlider', 'value')],
    # specify the component and corresponding properties that shall serve as input
    [dash.dependencies.State('input-on-submit',
                             'value')])  # specify the component and corresponding properties that shall serve as input
def update_output_train(n_clicks, range_n_estimators,range_min_samples_leaf, value):  # define the function reaching output from input
    if n_clicks % 2 is 1:
        BASE_PATH = value  # input value gives the base directory
        CODE_DIR = BASE_PATH + "spectrumQC/"
        DATA_DIR = BASE_PATH + "data/"

        TEMP_DIR = BASE_PATH + "temp/"
        MODEL_DIR = BASE_PATH + "model/"
        OUT_DIR = BASE_PATH + "output/"
        BIN_SIZE = 10

        os.chdir(CODE_DIR)
        data_dir = DATA_DIR
        temp_dir = TEMP_DIR
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        model_dir = MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        out_dir = OUT_DIR
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        bin_size = BIN_SIZE

        # prepare training set
        mzML_file_names = wrangling.mzMLfilename(data_dir)
        # parse mzML files to dictionary
        wrangling.mzML2dict(data_dir, temp_dir, bin_size)
        wrangling.evidenceDF(data_dir, temp_dir)
        training.trainingDataset(temp_dir, bin_size, mzML_file_names)
        lower_bound_n_estimators = range_n_estimators[0]
        upper_bound_n_estimators = range_n_estimators[1]
        lower_bound_min_samples_leaf = range_min_samples_leaf[0]
        upper_bound_min_samples_leaf = range_min_samples_leaf[1]

        param_grid = {'rf': {"min_samples_leaf": np.arange(lower_bound_min_samples_leaf,upper_bound_min_samples_leaf,2).tolist(), "min_samples_split": [5], "n_estimators": np.arange(lower_bound_n_estimators,upper_bound_n_estimators,10).tolist()},'svm': {"kernel": ['linear']}}
        final_result = training.modelling_spectrum_quality(temp_dir, model_dir, method='rf', param_grid=param_grid)
        X_train = final_result['X_train']
        y_train = final_result['y_train']
        X_test = final_result['X_test']
        y_test = final_result['y_test']
        final_model = final_result['model']
        feature_scores = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        fpr, tpr, thresholds = roc_curve(y_test, final_model.predict_proba(X_test)[:, 1])
        # feature_scores_df = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(
        #     ascending=False).reset_index()
        # fig = px.scatter(feature_scores_df, x="interval", y=0)
        figure_feature_importance = {'data':[dict(x=feature_scores.index.tolist(),y=feature_scores.tolist())],'layout': {'title': 'important features distribution'}}
        figure_roc_curve = {'data':[dict(x=fpr.tolist(),y=tpr.tolist())],'layout': {'title': 'roc curve'}}
        return figure_feature_importance,'AUC score of training process is "{}"'.format(roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])), figure_roc_curve,'Scope: "{}" '.format(range_n_estimators),'Scope: "{}" '.format(range_min_samples_leaf)
    else:
        return default_figure_feature_importance, ' ', default_roc_curve, 'Scope: "{}" '.format(range_n_estimators),'Scope: "{}" '.format(range_min_samples_leaf)


@app.callback(
    dash.dependencies.Output('train-indicator', 'color'),
    # specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-val', 'n_clicks')])
def start_train_indicator_button(n_clicks):
    if n_clicks % 2 is 1:
        value = 'red'
    else:
        value = 'grey'
    return value

@app.callback(
    dash.dependencies.Output('predict-indicator', 'color'),
    # specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-pred', 'n_clicks')])
def start_train_indicator_button(n_clicks):
    if n_clicks % 2 is 1:
        value = 'red'
    else:
        value = 'grey'
    return value



@app.callback(
    dash.dependencies.Output('spectrum quality prediction', 'children'),
    # specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-pred', 'n_clicks')],
    # specify the component and corresponding properties that shall serve as input
    [dash.dependencies.State('input-on-submit',
                             'value')]) # specify the component and corresponding properties that shall serve as input
def update_output_pred(n_clicks, value):  # define the function reaching output from input
    if n_clicks % 2 is 1:
        BASE_PATH = value  # input value gives the base directory

        CODE_DIR = BASE_PATH + "spectrumQC/"
        PREDICT_DIR = BASE_PATH + "predict/"

        MODEL_DIR = BASE_PATH + "model/"
        OUT_DIR = BASE_PATH + "output/"
        BIN_SIZE = 10

        os.chdir(CODE_DIR)
        predict_dir = PREDICT_DIR

        model_dir = MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        out_dir = OUT_DIR
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        bin_size = BIN_SIZE

        # prepare prediction set
        predictfile_names = wrangling.mzMLfilename(predict_dir)
        wrangling.mzML2dict(predict_dir, predict_dir, bin_size)
        prediction.predictDataset(predict_dir, bin_size, predictfile_names)

        # apply trained model for prediction
        prediction_result = prediction.predict_spectrum_quality(predict_dir, model_dir, out_dir)
        return 'prediction complete, please find the result in directory'
    else:
        return ''


# @app.callback(dash.dependencies.Output('intermediate-value-1', 'children'),
#               [dash.dependencies.Input('input-on-submit', 'value')])
@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    BASE_PATH = "F:/5500_P1_MVP/"  # input value gives the base directory
    OUT_DIR = BASE_PATH + "output/"

    return send_from_directory(OUT_DIR, path, as_attachment=True)


# @app.callback(dash.dependencies.Output('intermediate-value-2', 'children'),
#               [dash.dependencies.Input('input-on-submit', 'value')])
def uploaded_files():
    BASE_PATH = "F:/5500_P1_MVP/"  # input value gives the base directory
    OUT_DIR = BASE_PATH + "output/"

    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(OUT_DIR):
        path = os.path.join(OUT_DIR, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


# @app.callback(
#     dash.dependencies.Output("file-list", "children"),
#     [dash.dependencies.Input('input-on-submit', 'value')],
# )
# def update_output(value):
#     """Save uploaded files and regenerate the file list."""
#
#     files = uploaded_files()
#     if len(files) == 0:
#         return [html.Li("No files yet!")]
#     else:
#         return [html.Li(file_download_link(filename)) for filename in files]


if __name__ == '__main__':
    app.run_server(debug=True)