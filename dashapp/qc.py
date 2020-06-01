import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from dashapp.ui import parse_upload, get_spectrum_plot, get_svm_plot, get_full_ward_plot, get_dendrogram_plot


app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Mass Spec QA/QC"),
        dcc.Upload(
            html.Button("Upload Spectra"),
            id="upload-data",
            style={"float": "left"},
            multiple=True,
        ),
        dcc.Store(id="spectrum-data", storage_type="session"),
        html.H1("Spectrum vs Time Plot"),
        html.Div(id="xy-plot", style={"clear": "both"}),
        html.H1("SVM Plot"),
        html.Div(id="svm-plot", style={"clear": "both"}),
        html.H1("Clustergram Plot"),
        html.Div(id="dendrogram-plot", style={"clear": "both"}),
        html.Div(id="full-ward-plot", style={"clear": "both"}),
    ]
)

app.callback(
    Output("spectrum-data", "data"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)(parse_upload)

app.callback(Output("xy-plot", "children"), [Input("spectrum-data", "data")])(
    get_spectrum_plot
)

app.callback(Output("svm-plot", "children"), [Input("spectrum-data", "data")])(
    get_svm_plot
)

app.callback(Output("full-ward-plot", "children"), [Input("spectrum-data", "data")])(
    get_full_ward_plot
)

app.callback(Output("dendrogram-plot", "children"), [Input("spectrum-data", "data")])(
    get_dendrogram_plot
)