from re import template
import plotly.graph_objects as go

import pandas as pd
import numpy as np

# Read data from a csv
x_data_lin = np.linspace(-1,1,100)
y_data_lin = np.linspace(-1,1,100)
x_data,y_data = np.meshgrid(x_data_lin,y_data_lin)
z_data = -1.0/np.sqrt(x_data**2 + y_data**2)


z_min = -20
z_max = 0

fig = go.Figure(data=[go.Surface(z=z_data,cmin=z_min,cmax=z_max,x=x_data_lin, y=y_data_lin)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                highlightcolor="limegreen", project_z=True))
fig.update_layout(autosize=True,
                  scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                  scene = dict(zaxis=dict(range=[z_min,z_max])),
                  margin=dict(l=65, r=50, b=65, t=90), template='plotly_dark')
fig.update_coloraxes(cmin=z_min,cmax=z_max,reversescale=True,showscale=False)
fig.write_html("./neg_potential.html")
# fig.show()