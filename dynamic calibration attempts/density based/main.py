import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import webbrowser

import dash
from dash import dcc
from dash import html

from multiprocessing import Process

def f(fig):
    while True:
        fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 1)
        fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 1)
        fig.append_trace(go.Scatter3d(x=np.random.random(3), y=np.random.random(3), z=np.random.random(3)), 1, 1)
        
        
if __name__ == '__main__':
    
    fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

    fig.update_layout(scene_camera_eye=dict(x=2, y=2, z=2),
                      scene2_camera_eye=dict(x=2, y=2, z=2),
                      scene3_camera_eye=dict(x=2, y=2, z=2))
    
    p = Process(target=f, args=(fig,))
    p.start()
    
    app = dash.Dash()
    app.layout = html.Div([dcc.Graph(figure=fig)])
    
    webbrowser.open('http://127.0.0.1:8050/', new=2)
    app.run_server(debug=True, use_reloader=False)
    
    
    
    
    
    
    
    