#%%
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import itertools
from matplotlib import gridspec
from nilearn.surface import vol_to_surf

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
#Borrowed code from nilearn; adapted to change the grid layout to align spatial maps horizontally 1 x 4 
#Copyright (c) The nilearn developers.

def customized_plotting_img_on_surf(views, hemispheres, threshold=None, title=None, stat_map=None, vmin=None, vmax=None, output_file=None, cmap='cold_hot', colorbar=True, labels_left=None, labels_right=None, symmetric_cbar=True, label_color=None, axes=None):
    modes = plotting.surf_plotting._check_views(views=views)
    hemis = plotting.surf_plotting._check_hemispheres(hemispheres=hemispheres)
    surf_mesh = "fsaverage5"
    surf_mesh = plotting.surf_plotting._check_mesh(surf_mesh)

    inflate = False
    mesh_prefix = "infl" if inflate else "pial"

    surf = {
        "left": surf_mesh[mesh_prefix + "_left"],
        "right": surf_mesh[mesh_prefix + "_right"],
    }

    texture = {
        "left": vol_to_surf(stat_map, surf_mesh["pial_left"], mask_img=None),
        "right": vol_to_surf(stat_map, surf_mesh["pial_right"], mask_img=None),
    }
    gridspec_layout = gridspec.GridSpec(
        1, 4, left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0, wspace=0.0
    )


    fig = plt.figure(figsize=(40, 20))
    axes = []
    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):        
        bg_map = surf_mesh["sulc_%s" % hemi]
        ax = fig.add_subplot(gridspec_layout[i], projection="3d")
        axes.append(ax)
        fig1 =plotting.plot_surf_stat_map(
            surf[hemi],
            texture[hemi],
            view=mode,
            hemi=hemi,
            bg_map=bg_map,
            axes=ax, cmap=cmap,threshold=threshold,
            colorbar=colorbar, symmetric_cbar=symmetric_cbar,
            title=title, title_font_size=40, vmin= vmin, vmax=vmax, engine='matplotlib', output_file=output_file
        )
        if labels_right is not None:
            if hemi == 'right':
                parc=np.load(f"{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz")['labels_R']
                plotting.plot_surf_contours(surf[hemi], roi_map=parc, levels=labels_right, figure= fig1, axes=ax, color='blue')

        if labels_left is not None:
            if hemi == 'left':
            
                parc=np.load(f"{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz")['labels_L']
                plotting.plot_surf_contours(surf[hemi], roi_map=parc, levels=labels_left, figure= fig1, axes=ax, color='blue')

