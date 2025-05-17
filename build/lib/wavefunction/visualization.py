import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from .colormap import alpha_magma_r
from .wavefunction import PDFlize

def graphing_2D(psi, type=None, cmap='seismic', resol=0.07, list=None, size=12, title=None):
    """Plot a 2D orbital slice & slide (xz-plane)."""
    
    zmin=-size
    zmax=size
    dz=resol*zmax/4

    x = np.arange(zmin,zmax,dz)
    y = np.arange(zmin,zmax,dz)
    z = np.arange(zmin,zmax,dz)
    X,Y,Z = np.meshgrid(x,y,z)

    if list:
        data = psi(X,Y,Z,list)
    else:
        data = psi(X,Y,Z)

    if type == 'density':
        data = abs(data)**2
    else:
        data = data.real

    R = np.sqrt(X**2+Y**2+Z**2)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if type == 'density':
        vmin = 0
        vmax = np.max(data)
    else:
        vmin = -max(-np.min(data), np.max(data))
        vmax = max(-np.min(data), np.max(data))
    
    im = plt.imshow(data[int((0-zmin)/dz),:,:].T, vmin=vmin, vmax = vmax, extent=[zmin,zmax,zmin,zmax],cmap=cmap)
    plt.colorbar()
    sli = Slider(plt.axes([0.25, 0.025, 0.65, 0.03]), "Y", z[0], z[len(z)-1], valinit=0)

    if title is None and list is not None:
        Title = f"Hydrogen Orbital xz Slice"
    elif title is None:
        Title = f"Hydrogen Orbital xz Slice (y={str("%.2f"%sli.val)}): n={str(psi.n)}, l={str(psi.l)}, m={str(psi.m)}"
    else:
        Title=title
        
    ax.set_title(Title)

    def update(val):
        index = int((sli.val-zmin) / dz)
        im.set_data(data[index,:,:].T)
        ax.set_title(f"Hydrogen Orbital xz Slice (y={str("%.2f"%sli.val)}): n={str(psi.n)}, l={str(psi.l)}, m={str(psi.m)}")

    if title is None:    
        sli.on_changed(update)
        
    plt.show()

def graphing_3D(psi, num=500000, phi_limit=None, list=None, size=12, s=1, title=None):
    """Plot a 3D scatter of wavefunction density."""
        
    num_points = num
    s = s
    k = size

    if title is None and list is not None:
        title = f"3D Orbital Density"
    elif title is None:
        title = f"3D Orbital Density: n={psi.n}, l={psi.l}, m={psi.m}"
    else:
        title=title

    x = np.random.uniform(-k, k, num_points)
    y = np.random.uniform(-k, k, num_points)
    z = np.random.uniform(-k, k, num_points)

    if list:
        values = psi(x,y,z,list)
    else:
        values = psi(x,y,z)

    values = values**2

    if phi_limit:
        phi = np.arctan2(y, x)
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        mask_phi = (phi >= -(1/4)*np.pi) & (phi <= (6/4)*np.pi)

        x = x[mask_phi]
        y = y[mask_phi]
        z = z[mask_phi]
        values = values[mask_phi]

    norm_values = plt.Normalize(vmin=np.min(values), vmax=np.percentile(values, 99))
    mapped_colors = alpha_magma_r(norm_values(values))

    # 투명도가 0인 점을 없애기 위해 대단히 애먹었다!!!!
    alpha = mapped_colors[:, 3]
    mask = alpha > 0.15
    filtered_x = x[mask]
    filtered_y = y[mask]
    filtered_z = z[mask]
    filtered_values = values[mask]
    filtered_colors = mapped_colors[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(filtered_x, filtered_y, filtered_z, c=filtered_colors, s=s)

    sm = plt.cm.ScalarMappable(cmap=alpha_magma_r, norm=norm_values)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Probability Density')

    ax.set_title(title)
    ax.set_xlabel('X axis(Å)')
    ax.set_ylabel('Y axis(Å)')
    ax.set_zlabel('Z axis(Å)')
    ax.set_aspect("equal", adjustable="box")

    plt.show()