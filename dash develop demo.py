# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[ #.layout specifies what it looks like and what functions it is capable of 
    html.H1(children='tier 1 title: dash demo'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.H2(children='tier 2 title: a demo graph'),

    dcc.Graph( #dcc offers components fulfilling various kind of interactive visualization
        id='example: bar plot graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    ),
dcc.Dropdown( #add drop down menue; similarly we can add more
    options=[
        {'label': 'New York City', 'value': 'NYC'},
        {'label': 'Montreal', 'value': 'MTL'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    multi=True
)

])


if __name__ == '__main__':
    app.run_server(debug=True)