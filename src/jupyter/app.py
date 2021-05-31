# for future improvements with dash on the web
import numpy as np
import imageio
import matplotlib

from Trajectory import Trajectory2D, Trajectory3D

from matplotlib.animation import PillowWriter
from mpl_toolkits import mplot3d

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import pandas as pd

from collections import defaultdict
plt.rcParams.update({'font.size': 15})

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from IPython import embed

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)

'''
Re-arrenge data
Data points are always the same 20400
Rewards we have 10*400*51*1
Create dataframe with all coords repeated with respective similarity
and add extra column 'gs' indicating the goal state.
'''
goal_states = [
    [0,20,20]
]
initial_state = np.array([0, 0. ,  20], dtype=np.float32)

experiment = "VQVAE_CENTROIDS_1"
t = Trajectory3D(experiment, initial_state, goal_states, fix_lim=True)

rewards = t.rewards[0]
rewards = np.array(rewards).reshape(-1)
dlen = len(t.data_points[:,0])
data_dict = {'x': t.data_points[:,0], 'y': t.data_points[:,2], 'z': rewards, 'gs': np.zeros(dlen)}
df = pd.DataFrame.from_dict(data_dict)

for i in range(1, t.num_gstates):
    rewards = t.rewards[i]
    rewards = np.array(rewards).reshape(-1)
    data_dict = {'x': t.data_points[:,0], 'y': t.data_points[:,2], 'z': rewards, 'gs': np.ones(dlen)*i}
    df2 = pd.DataFrame.from_dict(data_dict)
    df = df.append(df2, ignore_index = True)



'''
For plotting the 3d scatterplot with dash it'll be smthg like this:
'''
app.layout = html.Div([
    dcc.Graph(id="scatter-plot"),
    html.P("Goal State:"),
    dcc.Slider(
        id='slider',
        min=0,
        max=t.num_gstates,
        step=1,
        value=0
    ),
])

@app.callback(
    Output("scatter-plot", "figure"),
    [Input("slider", "value")])
def update_figure(value):
    mask = (df.gs == value)
    fig = px.scatter_3d(df[mask].head(1000), x='x', y='y', z='z', color='z')
    fig.update_layout(transition_duration=500)
    return fig

app.run_server(debug=False)
