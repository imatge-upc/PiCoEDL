# for future improvements with dash on the web

'''
Re-arrenge data
Data points are always the same 20400
Rewards we have 10*400*51*1
Create dataframe with all coords repeated with respective similarity
and add extra column 'gs' indicating the goal state.
'''
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
    dcc.RangeSlider(
        id='range-slider',
        min=0, max=2.5, step=0.1,
        marks={0: '0', 2.5: '2.5'},
        value=[0.5, 2]
    ),
])

@app.callback(
    Output("scatter-plot", "figure"),
    [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df.petal_width > low) & (df.petal_width < high)

    fig = px.scatter_3d(df[mask],
        x='sepal_length', y='sepal_width', z='petal_width',
        color="species", hover_data=['petal_width'])
    return fig

app.run_server(debug=True)
