import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.constants import c, h, hbar, Boltzmann, elementary_charge

k_B = Boltzmann

# Create figure
fig = make_subplots(rows=1, cols=2)


# planck's law in wavelength
def I_wl(wl, T):

    # Calculate the exponential term
    exp_term = np.exp(h * c / (wl * k_B * T))

    # Apply Planck's Law formula
    intensity = (8 * np.pi * h * c**2 / (wl**5)) * (1 / (exp_term - 1))

    return intensity


# planck's law in wavelength
def I_nu(nu, T):

    # Calculate the exponential term
    exp_term = np.exp(h * nu / (k_B * T))

    # Apply Planck's Law formula (adapted for frequency)
    intensity = (8 * np.pi * h * nu**3 / (c**3)) * (1 / (exp_term - 1))

    return intensity


# Add traces, one for each slider step
v_wl = np.arange(10.0e-9, 6000e-9, 1e-9)
v_eV = np.linspace(0.0, 10, 1000)
v_E = v_eV * elementary_charge
v_nu = v_E / h

v_T = np.arange(50, 10000, 50)
for T in v_T:
    fig.add_trace(go.Scatter(visible=False,
                             line=dict(color="#000000", width=6),
                             name="Wavelength",
                             x=v_wl,
                             y=I_wl(v_wl, T)),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(visible=False,
                             line=dict(color="red", width=6),
                             name="Energy",
                             x=v_eV,
                             y=I_nu(v_nu, T)),
                  row=1,
                  col=2)
    fig.update_layout(plot_bgcolor='white',
                      font_size=30,
                      xaxis=dict(tickfont=dict(size=20),
                                 tickmode='array',
                                 tickvals=[500e-9 * i for i in range(1, 12)],
                                 ticktext=[str(500 * i) for i in range(1, 12)],
                                 title="Wavelength (nm)"),
                      yaxis=dict(tickfont=dict(size=20),
                                 title="Spectral Energy Density",
                                 showticklabels=False),
                      xaxis2=dict(
                          tickfont=dict(size=20),
                          tickmode='array',
                          tickvals=list(np.arange(1, 10, 1)),
                          ticktext=[str(eV) for eV in np.arange(1, 10, 1)],
                          title="Energy (eV)",
                      ),
                      yaxis2=dict(tickfont=dict(size=20),
                                  title="Spectral Energy Density",
                                  showticklabels=False))
    fig.update_xaxes(mirror=True,
                     ticks='outside',
                     showline=True,
                     linecolor='black',
                     gridcolor='lightgrey',
                     linewidth=6.0)
    fig.update_yaxes(mirror=True,
                     ticks='outside',
                     showline=True,
                     linecolor='black',
                     gridcolor='lightgrey',
                     linewidth=6.0)

# Make 10th trace visible
n_steps = len(fig.data) // 2 - 1
fig.data[n_steps - 1].visible = True
fig.data[n_steps].visible = True

# Create and add slider
steps = []
for i in range(n_steps):
    step = dict(
        method="update",
        label=str(v_T[i]),
        args=[{
            "visible": [False] * n_steps * 2
        }, {
            "title": "Temperature: " + str(v_T[i]) + " K"
        }],  # layout attribute
    )
    step["args"][0]["visible"][2 * i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][2 * i +
                               1] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [
    dict(
        active=n_steps // 2,
        currentvalue={
            "prefix": "T: ",
            'suffix': " K"
        },
        pad={"t": 50},
        steps=steps,
    )
]

fig.update_layout(sliders=sliders)

fig.write_html("./blackBody.html")
# fig.show()
