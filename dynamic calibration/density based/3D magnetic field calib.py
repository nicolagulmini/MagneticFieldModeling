import numpy as np
import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import cube_to_calib as CubeModel
from nidaq import NIDAQ
import CoilModel as Coil
from queue import Queue
import time
import pyigtl
import os
from flask import request

FILENAME = 'sampled_points'

DrfToAxis7 = np.array([
    [1,	0,					-0.0349154595874112,	-8.54323644559691],
    [0,	0.0213641235902639,	0.999162169684586,	1.90874592349076],
    [0,	0.999771761065104,	-0.0213510972315487,	-4.35779932552848],
    [0,	0,					0,					1]
    ])

channeldict = {0: 4, 1: 0, 2: 8, 3: 1, 4: 9, 5: 2, 6: 10, 7: 11, 8: 3, 9: 8, 10: 12, 11: 13, 12: 5, 13: 14, 14: 6, 15: 15, 16: 7}
sampleFreq = 40000
noSamples = 4000
freqs = np.array([6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400])
idx_signal = freqs/sampleFreq*noSamples+1
idx_signal = idx_signal.astype(int)
deviceID = 'Dev1'
sensor_channel = 7
sensor_channel = channeldict[sensor_channel]
PhaseOffset = 0

task = NIDAQ(dev_name = deviceID, channels = np.array([4, str(sensor_channel)]), sampleFreq = sampleFreq, data_len = noSamples)
task.SetAnalogInputs()
task.StartTask()

task1 = NIDAQ(dev_name=deviceID)
task1.SetClockOutput()
task1.StartTask()

AMOUNT_OF_NEW_POINTS = 10
print("Press the STOP button (or CTRL+C) when satisfied about the amount of gathered points. Suddenly the interpolation will be computed and the data will be stored in a .csv file.")

global q, cube, coil_model, client
q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
cube = CubeModel.cube_to_calib(np.array([-50., -50., 100.]), side_length=100., point_density=40., minimum_number_of_points=1)
coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)  

def get_fft(idx_signal):
    ft = task.get_data_matrix(timeout = 10.0)
    yf = np.fft.fft(ft, axis = 0) / noSamples
    yf = yf[idx_signal, :]
    return yf
    
def get_flux(yf, PhaseOffset):
    yf_mag = 2 * abs(yf[:, 1])
    yf_phase = np.angle(yf)
    angleSignal = yf_phase[:, 0] - yf_phase[:, 1] + PhaseOffset
    flux = np.sin(angleSignal) * yf_mag    
    return flux
    
def produce_basis_vectors_for_prediction(n):
    to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
    to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
    to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
    return to_pred_x, to_pred_y, to_pred_z

app = dash.Dash()
app.layout = html.Div(html.Div(children=[html.H1('Magnetic Field Freehand Calibration',
                                        style={'textAlign': 'center',
                                               'color': '#009BAC',
                                               'font-family': 'monospace'
                                               }
                                        ),
                                         html.Div(id='live-update-text'),
                                         dcc.Interval(id='interval-component',
                                                      interval = 25, # ms
                                                      n_intervals = 0),
                                         daq.StopButton(id='stop-button',
                                                        #label='',
                                                        n_clicks=0),
                                         html.Div(id='output of the button'),
                                         dcc.Graph(id='plot')
                                ]))

fig = make_subplots(horizontal_spacing=0.01, rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

color_scale = [[0., '#27FF00'], [.5, '#FFF700'], [1., '#FF2D00']]

c = np.zeros(cube.xline.shape[0])   

fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.3), 
                           name='x component', text='test'), 1, 1)
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.3), 
                           name='y component', text='test'), 1, 2)
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.3), 
                           name='z component', text='test'), 1, 3)
     
fig.add_trace(go.Cone(x=[cube.origin_corner[0]], y=[cube.origin_corner[1]], z=[cube.origin_corner[2]], u=[1], v=[0], w=[0], sizemode="absolute", sizeref=10, anchor="tip", showscale=False, colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']]), 1, 1)
fig.add_trace(go.Cone(x=[cube.origin_corner[0]], y=[cube.origin_corner[1]], z=[cube.origin_corner[2]], u=[0], v=[1], w=[0], sizemode="absolute", sizeref=10, anchor="tip", showscale=False, colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']]), 1, 2)
fig.add_trace(go.Cone(x=[cube.origin_corner[0]], y=[cube.origin_corner[1]], z=[cube.origin_corner[2]], u=[0], v=[0], w=[1], sizemode="absolute", sizeref=10, anchor="tip", showscale=False, colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']]), 1, 3)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=False, uirevision='_')
fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

if os.path.exists("./" + FILENAME + ".csv"):
    points = np.loadtxt("./" + FILENAME + ".csv")
    for i in range(int(points.shape[0]/10-1)):
        tmp_points = points[int(10*i):int(10*(i+1))]
        cube.add_batch(tmp_points)
    print('just loaded %.0f points' % (cube.points.shape[0]))
    perc_coverage_x, perc_coverage_y, perc_coverage_z = cube.percentages()
    print("Starting from coverage (c_x, c_y, c_z) = (%.2f, %.2f, %.2f)" % (perc_coverage_x, perc_coverage_y, perc_coverage_z))
    
    c_x, c_y, c_z = cube.contributions.T
    unc_x = 1.-np.minimum(c_x, np.ones(c_x.shape))
    unc_y = 1.-np.minimum(c_y, np.ones(c_x.shape))
    unc_z = 1.-np.minimum(c_z, np.ones(c_x.shape))
    
    # update markers' sizes and colors
    fig['data'][0]['marker']['color'] = unc_x
    fig['data'][0]['marker']['size'] = 20
    fig['data'][1]['marker']['color'] = unc_y
    fig['data'][1]['marker']['size'] = 20
    fig['data'][2]['marker']['color'] = unc_z
    fig['data'][2]['marker']['size'] = 20

client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)

message = client.wait_for_message("ReferenceToBoard", timeout=5)
referenceToBoard = message.matrix 

@app.callback(Output('plot', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n_intervals):
    
    if len(q.queue) < AMOUNT_OF_NEW_POINTS:
        
        # from the instrument
        message_1 = client.wait_for_message("SensorToReference", timeout=5)
        # print(message_1['Timestamp'])
        tmp = get_flux(get_fft(idx_signal), PhaseOffset)
        # print(idx_signal)
        message_2 = client.wait_for_message("SensorToReference", timeout=5)
        
        if message_1 is not None and message_2 is not None:
            
            mat_mul_1 = np.matmul(referenceToBoard, message_1.matrix)
            mat_1 = np.matmul(mat_mul_1, DrfToAxis7)
            
            mat_mul_2 = np.matmul(referenceToBoard, message_2.matrix)
            mat_2 = np.matmul(mat_mul_2, DrfToAxis7)
            
            pos = .5*(mat_1.T[3][:3]+mat_2.T[3][:3])
            ori = .5*(mat_1.T[2][:3]+mat_2.T[2][:3])
            
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
            
            if pos[0] >= cube.origin_corner[0] and pos[0] <= cube.origin_corner[0]+cube.side_length:
                if pos[1] >= cube.origin_corner[1] and pos[1] <= cube.origin_corner[1]+cube.side_length:
                    if pos[2] >= cube.origin_corner[2] and pos[2] <= cube.origin_corner[2]+cube.side_length:
                        q.put(np.concatenate((pos, ori, tmp), axis=0))
    
        return fig
        
    new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
    cube.add_batch(new_raw_points)

    # cube is global
    c_x, c_y, c_z = cube.contributions.T

    unc_x = 1.-np.minimum(c_x, np.ones(c_x.shape))
    unc_y = 1.-np.minimum(c_y, np.ones(c_y.shape))
    unc_z = 1.-np.minimum(c_z, np.ones(c_z.shape))

    # update markers' sizes and colors
    fig['data'][0]['marker']['color'] = unc_x
    fig['data'][0]['marker']['size'] = 20
    fig['data'][1]['marker']['color'] = unc_y
    fig['data'][1]['marker']['size'] = 20
    fig['data'][2]['marker']['color'] = unc_z
    fig['data'][2]['marker']['size'] = 20
        
    return fig

@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    perc_coverage_x, perc_coverage_y, perc_coverage_z = cube.percentages()
    perc_coverage_x, perc_coverage_y, perc_coverage_z = round(float(perc_coverage_x), 2), round(float(perc_coverage_y), 2), round(float(perc_coverage_z), 2)
    return [html.H2("coverage x: "+str(perc_coverage_x)+"%", style={'textAlign': 'left', 'color': '#009BAC', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '16px'}),
            html.H2("coverage y: "+str(perc_coverage_y)+"%", style={'textAlign': 'left', 'color': '#009BAC', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '16px'}),
            html.H2("coverage z: "+str(perc_coverage_z)+"%", style={'textAlign': 'left', 'color': '#009BAC', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '16px'}),
            ]

@app.callback(
    Output('output of the button', 'children'),
    Input('stop-button', 'n_clicks')
)
def update_output(n_clicks):
    if n_clicks >= 1:
        func = request.environ.get('werkzeug.server.shutdown')
        func() # it is deprecated but for now it works and idk how to do it in another way...
        return [html.H3('Saving %.0f points before quitting the calibration...' % cube.points.shape[0], style={'textAlign': 'left', 'color': '#FF2D00', 'font-family': 'monospace', 'padding': '5px', 'fontSize': '14px'})]

webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)

np.savetxt(FILENAME + ".csv", np.concatenate((cube.points, cube.measures), 1))
print('Just saved %.0f points' % (cube.points.shape[0]))