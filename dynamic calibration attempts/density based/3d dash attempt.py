import numpy as np
import plotly.graph_objects as go

import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from plotly.subplots import make_subplots

app = dash.Dash()
app.layout = html.Div(
    html.Div([
        html.H4('Magnetic Field Freehand Calibration'),
        # html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


# testo
# @app.callback(Output('live-update-text', 'children'),
#               Input('interval-component', 'n_intervals'))
# def update_metrics(n):
#     lon, lat, alt = 0., 0., 0.
#     style = {'padding': '5px', 'fontSize': '16px'}
#     return [
#         html.Span('Longitude: {0:.2f}'.format(lon), style=style),
#         html.Span('Latitude: {0:.2f}'.format(lat), style=style),
#         html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
#     ]


# plots
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    
    fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])
    
    fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 1)
    fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 2)
    fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 3)
    
    return fig


webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)