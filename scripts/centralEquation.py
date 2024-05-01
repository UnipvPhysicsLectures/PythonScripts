# --------------------------------- libraries ---------------------------------
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.constants as cs
import scipy 
from scipy import signal
from scipy.linalg import toeplitz
from cmath import phase
import matplotlib.colors as mcolors

# plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------- inputs ---------------------------------
n_k = 51     #number of k points in BZ
n_max = 10  # max number of reciprocal lattive vectors
a= 5*10**(-10)  #lattice constant
g = 2.0*np.pi / a
k_values = np.linspace(-g/2,g/2,n_k)
M = cs.m_e      #mass
dim = 512     # division in real space of the potential
x=np.linspace(-a/2,a/2,dim)  # x points of unit cell in real space 
V_max = 5.0
fill_fraction = 0.90
k_spring= 2.0*1e20

# --------------------------------- Functions ---------------------------------

# Potential
def square(x_arr, fill_f, V_maximum):
    """Generates a square wave potential in an array.

    Sets the first `fill_f` portion of the input array to `V_maximum` 
	and the remaining elements to 0. This creates a square wave potential with period 1.

    Args:
        x_arr (np.ndarray): The input array.
        fill_f (float): The fraction of elements to set to `V_maximum` (between 0 and 1).
		                This represents the 'high' portion of the square wave.
        V_maximum (float): The value to fill the 'high' portion of the square wave with.

    Returns:
        np.ndarray: The output array with a square wave potential.

    Raises:
        ValueError: If `fill_f` is not between 0 and 1.
    """

    dim_x = np.shape(x_arr)[0]
    V = np.zeros(dim_x)
    V[0:int(dim_x*fill_f)] = V_maximum
    return V

def harmonic(x_arr, k_spring):
	V = 0.5*k_spring*x_arr**2
	return V

def fft_V(V,n_max):
	"""Calculates the truncated real-valued Fast Fourier Transform (FFT) of a potential.

	Args:
	V (numpy.ndarray): The input signal to be transformed.
	n_max (int): The number of FFT coefficients to retain.

	Returns:
	numpy.ndarray: The truncated real-valued FFT of the input potential, 
				   with only the first 2*n_max coefficients preserved.
	"""
	V_fft = np.fft.rfft(V,norm='forward')
	V_fft[2*n_max+1:]=0
	return V_fft

def ifft_V(V_fft):
	"""
	Computes the inverse Fourier transform of a real-valued potential.

	Args:
	V_fft (numpy.ndarray): The input array containing the Fourier
							transform coefficients of the potential.

	Returns:
	numpy.ndarray: The reconstructed real-valued signal.
	"""
	iV_fft = np.fft.irfft(V_fft,norm='forward')
	return iV_fft

def V_GG(k,g,V_fft,n_max):
	"""Computes the coefficient matrix for the central equation.

	Args:
		k (float): The wavenumber in the Brillouin zone.
		g (float): The primitive reciprocal lattice vector.
		V_fft (array): The FFT of the potential.
		n_max (int): The maximum Fourier mode index.

	Returns:
		array: the central equation matrix.

	This function computes the coefficient matrix V_G for the central equation

	1. Computes the diagonal of the kinetic energy term using the formula
		`(hbar**2) * (k - n*g)**2 / (2*M*e)`, where `hbar` is the reduced Planck constant, 
		`M` is the electron mass, and `e` is the elementary charge.
	2. Constructs a Toeplitz matrix `V_G` from the first `2*n_max+1` elements of the 
		FFT of the potential and adds the diagonal kinetic energy term.

	This function assumes that the potential `V` is periodic with the periodicity of 
	the reciprocal lattice. The size of the FFT and the Toeplitz matrix are chosen 
	to be `2*n_max+1` to ensure accurate computation for all relevant k-points.

	Note that this function requires the following external libraries:

	- `numpy` for numerical operations and array manipulation.
	- `cs` (likely `constants`) for physical constants like `hbar`, `M`, and `e`.

	"""

	# compute the kinetic energy diagonal
	V_GG_diag = np.diag([(cs.hbar**2) * (k - n*g)**2/(2*M*cs.e) for n in np.arange(-n_max,n_max+1,1)])
	
	# compute the coeffiecient matrix and sum the kinetic diagonal
	
	# optional: setting to zero the zero frequency coefficient
	V_fft[0] = 0.0

	V_GG = toeplitz(V_fft[0:2*n_max+1])
	V_GG = V_GG + V_GG_diag
	return V_GG

def compute_bands(g,V_fft,n_max,n_k):
	"""Computes bands (eigenvalues) and wavefunction coefficients (eigenvectors)
	   for a given periodic potential.

	Args:
		g (float): The magnitude of the reciprocal lattice vector.
		V_fft (np.ndarray): The Fourier transform of the 1D  potential.
		n_max (int): The maximum Fourier mode index.
		n_k (int): The number of k-points to sample in the Brillouin zone.

	Returns:
		tuple: A tuple containing two numpy arrays:
			k_bands (np.ndarray, shape=(n_k, 2*n_max+1)): The array of eigenvalues for each k-point.
			k_wave_fft (np.ndarray, shape=(n_k, 2*n_max+1, 2*n_max+1)): The array of eigenvectors for each k-point.
	"""

	# building k sampling vector
	v_k = np.linspace(-g/2,g/2,n_k)

	# building storage matrixes for the eigenproblem
	m_E = np.zeros((n_k,2*n_max+1),dtype=np.complex128)
	m_ck = np.zeros((n_k,2*n_max+1,2*n_max+1),dtype=np.complex128)

	# solve the eigenproblem for all k values
	for i_k,k in enumerate(v_k):

		# compute the coefficients matrix
		m_GG = V_GG(k,g,V_fft,n_max)

		# solve the eigenproblem for the given k value
		m_E[i_k],m_ck[i_k] = np.linalg.eigh(m_GG, UPLO='U')

	return m_E, m_ck


def compute_wavefunction(g,x,m_ck,i_k,i_band,n_max,n_k):
    """Computes the wavefunction for a given band and k-point.

    This function explicitly calculates the wavefunction for a specified band and k-point within a user-defined k-point sampling scheme.

    Args:
        g (float): Elementary reciprocal lattive vector
        x (np.ndarray): The position array.
        m_ck (np.ndarray): Array of coefficients for the wavefunction.
        i_k (int): Index of the k-point in `m_ck` to use for calculation.
        i_band (int): Index of the band in `m_ck`.
        n_max (int): Maximum value for summation in the wavefunction calculation.
        n_k (int): Number of k-points used for sampling. The function internally builds
		           a wavevector array `v_k` of size `n_k` ranging from -g/2 to g/2.

    Returns:
        np.ndarray: The wavefunction for the specified band and k-point as a complex array.
    """
    # building k sampling vector
    v_k = np.linspace(-g/2,g/2,n_k)

    # select k value
    k = v_k[i_k]

    # explicitly compute the wavefunction
    wavefunct = np.zeros(len(x), dtype=np.complex128)
    for i_c, c_k in zip(np.arange(-n_max,n_max+1,1),m_ck[i_k,:,i_band]):
        wavefunct = wavefunct + c_k*np.exp( 1j*(k+i_c*g)*x)

    return wavefunct

# --------------------------------- Solution ---------------------------------
V = square(x,fill_fraction,2*V_max)
# V = harmonic(x,k_spring)
V_fft = fft_V(V,n_max)
rV = ifft_V(V_fft)
v_k = np.linspace(-g/2,g/2,n_k)
k_min, k_max = v_k.min(), v_k.max()
m_E, m_ck = compute_bands(g,V_fft,n_max,n_k)
wavefunct_val = compute_wavefunction(g,x,m_ck,0,0,n_max,n_k)
wavefunct_cond = compute_wavefunction(g,x,m_ck,0,1,n_max,n_k)

# repeated potentials
tile_x = np.concatenate((x-a,x,x+a))
tile_V = np.tile(V,5)
tile_rV = np.tile(rV,5)

# --------------------------------- Plotting  ---------------------------------
# Create figure
fig = make_subplots(rows=1, cols=2)
colors = [ '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

# add the first 4 bands to the plot
for i in range(4):
    fig.add_trace(go.Scatter(visible=True,
                                line=dict(color=colors[i], width=3),
                                name="Band n="+str(i),
                                x=v_k/g,
                                y=np.real(m_E[:,i])),
                    row=1,
                    col=1)

# plot the potential to the right
fig.add_trace(go.Scatter(visible=True,
                            line=dict(color="black", width=3),
                            name="Periodic Potential",
                            x=tile_x/a,
                            y=tile_V),
                row=1,
                col=2)
fig.add_trace(go.Scatter(visible=True,
                            line=dict(color="blue", width=3,dash='dot'),
                            name="Reconstructed Potential",
                            x=tile_x/a,
                            y=tile_rV),
                row=1,
                col=2)

# loop over all the k values to add hidden traces
for i_k, k in enumerate(v_k):

    # add the sliding dots
    for i in range(2):
        fig.add_trace(go.Scatter(visible=False,
                                mode='markers',
                                marker=dict(color=colors[i], size=10),
                                name=r"$\large \mathrm{i_{k}=" + str(i_k) + "}$",
                                x=[k/g],
                                y=[np.real(m_E[i_k,i])]),
                    row=1,
                    col=1)

    # compute wavefunctions for appropriate
    wavefunct_val = compute_wavefunction(g,x,m_ck,i_k,0,n_max,n_k)
    wavefunct_cond = compute_wavefunction(g,x,m_ck,i_k,1,n_max,n_k)

    # repeated wavefunctions
    tile_val = np.tile(wavefunct_val,5)
    tile_cond = np.tile(wavefunct_cond,5)

    fig.add_trace(go.Scatter(visible=False,
                        marker=dict(color=colors[0], size=10),
                        name=r"$\large \mathrm{\psi_{val}(x)}$",
                        x=(tile_x/a-1/2),
                        y=np.abs(tile_val)**2),
            row=1,
            col=2)
    fig.add_trace(go.Scatter(visible=False,
                        marker=dict(color=colors[1], size=10),
                        name=r"$\large \mathrm{\psi_{cond}(x)}$",
                        x=(tile_x/a-1/2),
                        y=np.abs(tile_cond)**2),
            row=1,
            col=2)


fig.update_layout(plot_bgcolor='white',
                    font_size=20,
                    xaxis=dict(tickfont=dict(size=20),
                                # tickmode='array',
                                # tickvals=[i for i in range(-6, 6)],
                                # ticktext=[str(500 * i) for i in range(1, 12)],
                                title="Normalized Crystal Momentum (k/g)",
                                range=[-0.5,0.5]),
                    yaxis=dict(tickfont=dict(size=20),
                                title="E (eV)",
                                showticklabels=True),
                    xaxis2=dict(
                        tickfont=dict(size=20),
                        tickmode='array',
                        # tickvals=list(np.arange(1, 10, 1)),
                        # ticktext=[str(eV) for eV in np.arange(1, 10, 1)],
                        title="Normalized Position (x/a)",
                        range=[-1,1]),
                    yaxis2=dict(tickfont=dict(size=20),
                                # title="Spectral Energy Density",
                                showticklabels=True))
fig.update_xaxes(mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    linewidth=3.0)
fig.update_yaxes(mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    linewidth=3.0)

# Make edge state visible
fig.data[6].visible = True
fig.data[6 + 1].visible = True
fig.data[6 + 2].visible = True
fig.data[6 + 3].visible = True



# Create and add slider
steps = []
for i in range(n_k):
    visibility_list = [False] * (n_k * 4 + 6)
    for n in range(6):
        visibility_list[n] = True
    step = dict(
        method="update",
        label=str(np.round(v_k[i]/g,2)),
        args=[{
            "visible": visibility_list
        }, {
            "title": "k/g: " + str(np.round(v_k[i]/g,2))
        }],  # layout attribute
    )

    # Toggle i'th trace to "visible"
    step["args"][0]["visible"][4 * i + 6 ] = True  
    step["args"][0]["visible"][4 * i + 6 + 1] = True
    step["args"][0]["visible"][4 * i + 6 + 2] = True  
    step["args"][0]["visible"][4 * i + 6 + 3] = True
    steps.append(step)

sliders = [
    dict(
        active=0,
        currentvalue={
            "prefix": "k/g: ",
        },
        pad={"t": 50},
        steps=steps,
    )
]

fig.update_layout(sliders=sliders)
fig.write_html("../html/centralEquation.html")
# fig.show()