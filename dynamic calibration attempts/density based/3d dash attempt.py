import numpy as np
import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import cube_to_calib as CubeModel
import CoilModel as Coil
from queue import Queue
import time

AMOUNT_OF_NEW_POINTS = 50

global q, cube, coil_model
q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
cube = CubeModel.cube_to_calib(np.array([-25., 25., 5.]), side_length=50., point_density=10., minimum_number_of_points=5)
coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)  


app = dash.Dash()
app.layout = html.Div(html.Div(children=[html.H1('Magnetic Field Freehand Calibration',
                                        style={'textAlign': 'center',
                                               'color': '#009BAC',
                                               'font-family': 'monospace'
                                               }
                                        ),
                                html.Div(id='live-update-text'),
                                dcc.Graph(id='plot'),
                                dcc.Interval(id='interval-component',
                                             interval = 1000, # ms
                                             n_intervals = 0)
                                ]))

fig = make_subplots(horizontal_spacing=0.01, rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

color_scale = [[.0, '#27FF00'], [.5, '#FFF700'], [1.0, '#FF2D00']]

c = np.zeros(cube.xline.shape[0])

# initialization
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.3), 
                           name='x component', text='test'), 1, 1)
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.3), 
                           name='y component', text='test'), 1, 2)
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.3), 
                           name='z component', text='test'), 1, 3)

# fig.add_trace(ff.create_quiver(), 1, 1)

# pos_sensor = np.flip(new_raw_points[:, :6], 0)[:, :3]
# or_sensor = np.flip(new_raw_points[:, :6], 0)[:, 3:]

# ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
# ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
# az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
      
random_quiver_pos = cube.origin_corner + cube.side_length*np.random.random(3)
random_quiver_pos = random_quiver_pos[:,np.newaxis]

fig.add_trace(go.Cone(x=[cube.origin_corner[0]], y=[cube.origin_corner[1]], z=[cube.origin_corner[2]], u=[1], v=[0], w=[0], sizemode="absolute", sizeref=10, anchor="tip", showscale=False))
fig.add_trace(go.Cone(x=[cube.origin_corner[0]], y=[cube.origin_corner[1]], z=[cube.origin_corner[2]], u=[0], v=[1], w=[0], sizemode="absolute", sizeref=10, anchor="tip", showscale=False))
fig.add_trace(go.Cone(x=[cube.origin_corner[0]], y=[cube.origin_corner[1]], z=[cube.origin_corner[2]], u=[0], v=[0], w=[1], sizemode="absolute", sizeref=10, anchor="tip", showscale=False))

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=False, uirevision='_')
fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

print(fig)

@app.callback(Output('plot', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n_intervals):
    
    while len(q.queue) < AMOUNT_OF_NEW_POINTS:
        pos, ori = cube.origin_corner + cube.side_length*np.random.random(3), np.random.random(3)
        
        fig['data'][3]['x'] = [pos[0]]
        fig['data'][3]['y'] = [pos[1]]
        fig['data'][3]['z'] = [pos[2]]
        fig['data'][3]['u'] = [ori[0]]
        fig['data'][3]['v'] = [ori[1]]
        fig['data'][3]['w'] = [ori[2]]
        
        fig['data'][4]['x'] = [pos[0]]
        fig['data'][4]['y'] = [pos[1]]
        fig['data'][4]['z'] = [pos[2]]
        fig['data'][4]['u'] = [ori[0]]
        fig['data'][4]['v'] = [ori[1]]
        fig['data'][4]['w'] = [ori[2]]
        
        fig['data'][5]['x'] = [pos[0]]
        fig['data'][5]['y'] = [pos[1]]
        fig['data'][5]['z'] = [pos[2]]
        fig['data'][5]['u'] = [ori[0]]
        fig['data'][5]['v'] = [ori[1]]
        fig['data'][5]['w'] = [ori[2]]        
        
        tmp = get_theoretical_field(coil_model, pos, ori)
        q.put(np.concatenate((pos, ori, tmp.A1), axis=0))
    new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
    cube.add_batch(new_raw_points)

    # cube is global
    c_x, c_y, c_z = cube.contributions.T

    unc_x = 1.-np.minimum(c_x, np.ones(c_x.shape))
    unc_y = 1.-np.minimum(c_y, np.ones(c_x.shape))
    unc_z = 1.-np.minimum(c_z, np.ones(c_x.shape))

    # update markers' sizes and colors
    fig['data'][0]['marker']['color'] = unc_x
    fig['data'][0]['marker']['size'] = 10
    fig['data'][1]['marker']['color'] = unc_y
    fig['data'][1]['marker']['size'] = 10
    fig['data'][2]['marker']['color'] = unc_z
    fig['data'][2]['marker']['size'] = 10
        
    return fig

@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    c_x, c_y, c_z = cube.contributions.T 
    perc_coverage_x = round(float(sum(np.minimum(c_x, np.ones(c_x.shape)))/c_x.shape)*100, 2)
    perc_coverage_y = round(float(sum(np.minimum(c_y, np.ones(c_y.shape)))/c_y.shape)*100, 2)
    perc_coverage_z = round(float(sum(np.minimum(c_z, np.ones(c_z.shape)))/c_z.shape)*100, 2)
    return [html.H2("coverage x: "+str(perc_coverage_x)+"%", style={'textAlign': 'left', 'color': '#009BAC', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '16px'}),
            html.H2("coverage y: "+str(perc_coverage_y)+"%", style={'textAlign': 'left', 'color': '#009BAC', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '16px'}),
            html.H2("coverage z: "+str(perc_coverage_z)+"%", style={'textAlign': 'left', 'color': '#009BAC', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '16px'}),
            ]

webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)