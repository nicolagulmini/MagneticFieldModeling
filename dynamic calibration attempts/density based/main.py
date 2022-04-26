import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import numpy as np

import webbrowser

import time

import dash
from dash import dcc
from dash import html

import threading

def f(fig):
    while True:
        fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 1)
        fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 2)
        fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 3)
        time.sleep(1)
        
def app_thread(app):
    app.run_server(debug=True, use_reloader=False)
    
def reload():
    return html.Div([dcc.Graph(figure=fig)])
        
if __name__ == '__main__':
    
    fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

    fig.update_layout(scene_camera_eye=dict(x=2, y=2, z=2),
                      scene2_camera_eye=dict(x=2, y=2, z=2),
                      scene3_camera_eye=dict(x=2, y=2, z=2))
    
    t = threading.Thread(target=f, args=[fig,])
    t.start()
    
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
    
    webbrowser.open('http://127.0.0.1:8050/', new=2)
    
    t2 = threading.Thread(target=app_thread, args=[app,])
    t2.start()
    
    @app.callback(Output('live-update-graph', 'figure'),
                  Input('interval-component', 'n_intervals'))
    def update_graph_live(n):
        return fig
    
    while True:
        app.layout = reload
        time.sleep(1)