import wrangling, prediction, training, constants
import os
import dash
import dash_html_components as html
import dash_core_components as dcc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(dcc.Input(id='input-on-submit', type='text',value='F:/5500_P1_MVP/')), #add an input bar
    html.Button('Submit and start training process', id='submit-val', n_clicks=0), #add a button to start training
    html.Div(id='spectrum quality prediction', #add a section to store and display output
             children='Please enter the file directory')
])


@app.callback(
    dash.dependencies.Output('spectrum quality prediction', 'children'), #specify the component and its property that shall contain the output
    [dash.dependencies.Input('submit-val', 'n_clicks')], #specify the component and corresponding properties that shall serve as input
    [dash.dependencies.State('input-on-submit', 'value')]) #specify the component and corresponding properties that shall serve as input
def update_output(n_clicks, value): #define the function reaching output from input
    BASE_PATH = value #input value gives the base directory
    CODE_DIR = BASE_PATH + "spectrumQC/"
    DATA_DIR = BASE_PATH + "data/"
    PREDICT_DIR = BASE_PATH + "predict/"

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


if __name__ == '__main__':
    app.run_server(debug=True)