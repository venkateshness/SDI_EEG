#%%
import sklearn.preprocessing
from tqdm import tqdm
import numpy as np
import utility_functions
import matplotlib.pyplot as plt
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"

condition = 'rest'
band = "widerband"

#%%
graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")
envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/{condition}/cortical_surface_related/parcellated_widerband.npz")

for sub, signal in tqdm(envelope_signal_bandpassed.items()):
    _, eigenvals, eigenvectors = utility_functions.eigmodes(W = graph[sub])

# %%

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def compute_gap_spectrum(eigenvalues):
    """Fit a spline to eigenvalues and compute the second derivative."""
    x = np.linspace(0, len(eigenvalues)-1, len(eigenvalues))
    spline = UnivariateSpline(x, eigenvalues, k=5, s=3)
    second_derivative = spline.derivative(n=2)(x)
    return second_derivative

def find_critical_points(second_derivative):
    """Identify the indices of maxima and minima in the 2nd order gap spectrum."""
    zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]
    print(zero_crossings)
    return zero_crossings

def plot_gap_spectrum(eigenvalues, second_derivative, title='2nd Order Gap Spectrum'):
    """Plot eigenvalues and their second derivative."""
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(eigenvalues, label='Eigenvalues')
    plt.title('Eigenvalues')
    plt.subplot(122)
    plt.plot(second_derivative, label='2nd Order Derivative')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example data: Simulated eigenvalues for 3 subjects
all_critical_points = []

for sub, signal in tqdm(envelope_signal_bandpassed.items()):
    
    _, eigenvals, eigenvectors = utility_functions.eigmodes(W = graph[sub])
    second_derivative = compute_gap_spectrum(eigenvals)
    
    zero_crossings = find_critical_points(second_derivative)
    
    all_critical_points.append(zero_crossings)
    plot_gap_spectrum(eigenvals, second_derivative)


# %%
all_critical_points
# %%
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Generate example eigenvalues (Replace this with your actual data)
np.random.seed(42)  # For reproducibility
eigenvalues = eigenvals

# Define the indices for the eigenvalues
x_indices = np.arange(len(eigenvalues))

# Fit a spline to the eigenvalues with order 10 and 3 internal knots
knots = np.quantile(eigenvalues, [0.25, 0.5, 0.75])
spline = UnivariateSpline(x=x_indices, y=eigenvalues, k=5, s=3)

# Generate a dense range of points to evaluate the spline and make a smooth curve
x_dense = np.linspace(0, len(eigenvalues) - 1, 360)
spline_values = spline(x_dense)

# Calculate RMSE to quantify the fit error
spline_approximation = spline(x_indices)  # Evaluate spline at the original data points
rmse = np.sqrt(np.mean((spline_approximation - eigenvalues) ** 2))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_indices, eigenvalues, 'o', label='Original Eigenvalues')
plt.plot(x_dense, spline_values, label='Spline Fit', linewidth=2)
plt.title(f'Spline Fit Validation (RMSE: {rmse:.4f})')
plt.legend()
plt.grid(True)
plt.show()

# %%


np.argwhere(second_derivative<0)
# %%
