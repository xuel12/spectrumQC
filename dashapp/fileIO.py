import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from subprocess import call
from dash.exceptions import PreventUpdate

from wrangling import evidenceDF

UPLOAD_DIRECTORY = "/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/temp1"
DATA_DIR = "/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/data"
TEMP_DIR = "/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/temp"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div(
    [
        html.H1("File Browser"),
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        dcc.Input(
            id = 'data-dir',
            placeholder='Data directory',
            type='text',
            value='',
        ),  # fill out your Input however you need
        dcc.Input(
            id = 'temp-dir',
            placeholder='Temp directory',
            type='text',
            value='',
        ),  # fill out your Input however you need
        html.Div(
            html.Button('Predict', id='button'),
            html.Div(id='output-container-button',
                    children=['Hit the button to update.']),
        ),
        html.H2("File List"),
        html.Ul(id="file-list"),
    ],
    style={"max-width": "500px"},
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
# @app.callback(
#     Output("file-list", "children"),
#     [Input("data-dir", "value"), Input("temp-dir", "value")],
# )
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]

@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')])
def run_script_onClick(n_clicks):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate

    script_path = 'python /Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/spectrumQC/prediction.py'
    # The output of a script is always done through a file dump.
    # Let's just say this call dumps some data into an `output_file`
    call(["python3", script_path])

    # # Load your output file with "some code"
    # output_content = some_loading_function('output file')

    # Now return.
    # return output_content
    return 1

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)