import numpy as np

# Bohr radius in angstroms
a0 = 0.529177210544

def transmeter(a, type='nmtoA'):
    """Convert units between nanometers, angstroms and Bohr radius."""
    if type == 'nmtoA':
        return a * 10
    elif type == 'Atonm':
        return a / 10
    elif type == 'AtoBohr':
        return a / a0
    elif type == 'BohrtoA':
        return a * a0
    elif type == 'nmtoBohr':
        return transmeter(transmeter(a, 'nmtoA'), 'AtoBohr')
    elif type == 'Bohrtonm':
        return transmeter(transmeter(a, 'BohrtoA'), 'Atonm')
    else:
        raise ValueError("Invalid conversion type. Use 'nmtoA' or 'Atonm' or 'AtoBohr' or 'BohrtoA'.")