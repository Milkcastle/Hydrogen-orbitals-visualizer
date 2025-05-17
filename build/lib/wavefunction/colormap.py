from matplotlib.colors import LinearSegmentedColormap

# Create a modified magma colormap with transparency
colors = [
    (1, 1, 1, 0), #0.0
    (1, 0.85, 0, 0.16), #0.15
    (1, 0.5, 0, 0.32), #0.25
    (1, 0, 0, 0.48), #0.5
    (0.5, 0, 0.5, 0.64), #0.5
    (0, 0, 0.5, 0.80), #0.75
    (0, 0, 0, 0.96), #1.0
]
n_bins = 1000
cmap_name = 'alpha_magma_r'
alpha_magma_r = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)