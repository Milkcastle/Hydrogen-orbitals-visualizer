import numpy as np
from scipy.special import genlaguerre, sph_harm
from .constants import a0
import math

def PSI_nlm(x, y, z, n, l, m):
    """Calculate the hydrogen atom orbital wavefunction (complex-valued)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, where=r!=0))
    phi = np.arctan2(y, x)
    
    rho = 2 * r / (n * a0)
    norm_C = np.sqrt((2 / (n * a0))**3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    R_nl = norm_C * np.exp(-rho / 2) * rho**l * genlaguerre(n - l - 1, 2 * l + 1)(rho)
    Y_lm = sph_harm(m, l, phi, theta)
    return R_nl * Y_lm

def make_psi(n, l, m):
    def psi_real(x, y, z):
        """Real-valued hydrogenic orbital function."""
        if m < 0:
            psi_real = (1j/np.sqrt(2))*(PSI_nlm(x, y, z, n, l, -np.abs(m))-((-1)**m)*(PSI_nlm(x, y, z, n, l, np.abs(m))))
        elif m == 0:
            psi_real = PSI_nlm(x, y, z, n, l, 0)
        elif m > 0:
            psi_real = (1/np.sqrt(2))*(PSI_nlm(x, y, z, n, l, -np.abs(m))+((-1)**m)*(PSI_nlm(x, y, z, n, l, np.abs(m))))
        return psi_real
    
    psi_real.n = n
    psi_real.l = l
    psi_real.m = m
    return psi_real

def PSI_real(x, y, z, n, l, m):
    psi = make_psi(n, l, m)
    return psi(x, y, z)
    
orbitals = [(2, 0, 0), (2, 1, -1), (2, 1, 1), (2, 1, 0), (3, 2, 0), (3, 2, 2)]
psis = [make_psi(n, l, m) for (n, l, m) in orbitals]

def hybrid_psi(x, y, z, coeffs):
    return sum(c * psi(x, y, z) for c, psi in zip(coeffs, psis))

# coeffs example
''' 
[np.sqrt(1/6), 0, 0, np.sqrt(2/3), np.sqrt(2/3), 0] # sp3d2
[np.sqrt(1/2), np.sqrt(1/2), 0, 0, 0, 0] # sp
[np.sqrt(1/3), np.sqrt(1/6), np.sqrt(1/2), 0, 0, 0] # sp2
[np.sqrt(1/4), np.sqrt(1/4), np.sqrt(1/4), np.sqrt(1/4), 0, 0] # sp3
[np.sqrt(1/5), 0, 0, np.sqrt(3/10), np.sqrt(2/5), 0] # sp3d
[np.sqrt(1/6), np.sqrt(2/3), 0, 0, 0, np.sqrt(1/6)] # sp3d2
'''

def PDFlize(psi):
    """Convert wavefunction to probability density function (PDF)."""
    return np.abs(psi)**2
