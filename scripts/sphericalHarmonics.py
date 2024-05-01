import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.special import sph_harm

# define quantum spherical harmonics
def Yml(theta,phi,m,l):

    if m >=0:
        Ymn = (-1)**m * sph_harm(m,l,phi,theta)
    else:
        Ymn = sph_harm(-m,l,phi,theta)

    return np.abs(Ymn),np.angle(Ymn)

# Equation of ring cyclide
# see https://en.wikipedia.org/wiki/Dupin_cyclide
import numpy as np
theta, phi = np.mgrid[0:np.pi:100j, 0:2*np.pi:200j]

l = 2
m = 0
Ymn, Ymn_phase = Yml(theta,phi,m,l)
x = Ymn*np.sin(theta)*np.cos(phi)
y = Ymn*np.sin(theta)*np.sin(phi)
z = Ymn*np.cos(theta)


fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['Color corresponds to distance from origin', 'Color corresponds to phase'],
                    )

fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=Ymn, colorbar_x=-0.07,colorbar={"title": 'Your title'}), 1, 1)
fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=Ymn_phase,colorscale='RdBu'), 1, 2)
fig.update_layout(title_text="Spherical Harmonics")
fig.show()