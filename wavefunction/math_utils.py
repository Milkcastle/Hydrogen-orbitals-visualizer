import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .constants import a0
import math

def diff_exp(func, v, p):
    """Return the p-th derivative of a symbolic expression with respect to variable v."""
    for _ in range(p):
        func = sp.diff(func, v)
    return func

def diff_poly(poly, n):
    """Return the n-th derivative of entered polynomial"""
    result_poly = poly
    for _ in range(n):
        result_poly = result_poly.deriv()
    return result_poly

def EQtoEX(eq):
    """Convert symbolic equation to numpy-evaluable lambda function."""
    x = sp.symbols('x')
    return sp.lambdify(x, eq, 'numpy')

def special_1(func_str, p):
    """Evaluate the p-th derivative of polynomial expressed as string."""
    x = sp.symbols('x')
    f = parse_expr(func_str)
    for _ in range(p):
        f = sp.diff(f, x)
    return ((-1)**p) * f

def special_2(func_str, p):
    """Evaluate the p-th derivative with alternating sign."""
    x = sp.symbols('x')
    f = parse_expr(func_str)
    for _ in range(p):
        f = sp.diff(f, x)
    return ((-1)**p) * f

def P_ml(x, m, l):
    """Evaluate the general legendre equation."""
    poly = np.poly1d([1, 0, -1])
    legendre = (1/((2**l)*(math.factorial(l)))) * diff_poly(poly**l, l)
    func = ((-1)**abs(m))*((1-(x**2))**(abs(m)/2))*(diff_poly(legendre, abs(m))(x))
    if m < 0:
        func = ((-1)**abs(m))*(math.factorial((l-abs(m)))/math.factorial((l+abs(m))))*P_ml(x, abs(m), l)
    return func

def Y_ml(m, l, theta, phi):
    """Evaluate the spherical harmonics function."""
    func = abs((np.sqrt(((2*l+1)*math.factorial((l-m)))/((4*np.pi)*math.factorial((l+m)))))*P_ml(np.cos(theta), m, l))*((np.exp(1j * m * phi)))
    return func

def L_pq(x, p, q):
    """Evaluate the general laguarre equation."""
    L_q = special_1(x, p, q)
    func = special_2(str(L_q), x, p)
    return func

def R_nl(n, l, r):
    """Evaluate the radial wavefunction."""
    func = (np.sqrt(((2/(n*a0))**3)*(math.factorial(n-l-1)/(2*n*(math.factorial(n+l))))))*(np.exp(-r/(n*a0)))*(((2*r)/(n*a0))**l)*L_pq((2*r)/(n*a0))
    return func