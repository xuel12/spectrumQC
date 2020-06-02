import wrangling, prediction, training, constants
import os
from urllib.parse import quote as urlquote

import dash
import dash_html_components as html
import dash_core_components as dcc
from flask import Flask, send_from_directory

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.H1("Mass spec spectrum QC"),
    html.H4("Please specify the base directory"),
    # html.Div(dcc.Input(id='input-on-submit', type='text', value='F:/5500_P1_MVP/')), #add an input bar
    html.Div(dcc.Input(id='input-on-submit', type='text', value='/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/')), #add an input bar
    html.H2("Training"),
    html.Button('Start training process', id='submit-val', n_clicks=0), #add a button to start training
    html.Div(id='spectrum quality training'), #add a section to store and display output
    html.H2("Prediction"),
    html.Button('Start prediction process', id='submit-pred', n_clicks=0), #add a button to start training
    html.Div(id='spectrum quality prediction'), #add a section to store and display output
    html.H2("File List"),
    html.Ul(id="file-list"),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value-1', style={'display': 'none'}),
    html.Div(id='intermediate-value-2', style={'display': 'none'})

])


@app.callback(
    dash.dependencies.Output('spectrum quality training', 'children'), #specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-val', 'n_clicks')], #specify the component and corresponding properties that shall serve as input
    [dash.dependencies.State('input-on-submit', 'value')]) #specify the component and corresponding properties that shall serve as input
def update_output_train(n_clicks, value): #define the function reaching output from input
    if n_clicks!=0:
        BASE_PATH = value #input value gives the base directory
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
    
        param_grid = {'rf': {"min_samples_leaf": [2], "min_samples_split": [5], "n_estimators": [50]}, \
                      'svm': {"kernel": ['linear']}}
        model = training.modelling_spectrum_quality(temp_dir, model_dir, method='rf', param_grid=param_grid)
        return 'AUC score of training process is "{}"'.format( #return AUC score to
            model.score()
        )

@app.callback(
    dash.dependencies.Output('spectrum quality prediction', 'children'), #specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-pred', 'n_clicks')], #specify the component and corresponding properties that shall serve as input
    [dash.dependencies.State('input-on-submit', 'value')]) #specify the component and corresponding properties that shall serve as input
def update_output_pred(n_clicks, value): #define the function reaching output from input
    if n_clicks!=0:
        BASE_PATH = value #input value gives the base directory
        
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
        return prediction_result

# @app.callback(dash.dependencies.Output('intermediate-value-1', 'children'), 
#               [dash.dependencies.Input('input-on-submit', 'value')])
@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    BASE_PATH = '/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/' #input value gives the base directory
    OUT_DIR = BASE_PATH + "output/"

    return send_from_directory(OUT_DIR, path, as_attachment=True)

        
        
# @app.callback(dash.dependencies.Output('intermediate-value-2', 'children'), 
#               [dash.dependencies.Input('input-on-submit', 'value')])
def uploaded_files():
    BASE_PATH = '/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/' #input value gives the base directory
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


@app.callback(
    dash.dependencies.Output("file-list", "children"),
    [dash.dependencies.Input('input-on-submit', 'value')],
)
def update_output(value):
    """Save uploaded files and regenerate the file list."""

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]
    


if __name__ == '__main__':
    app.run_server(debug=True)