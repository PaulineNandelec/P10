#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import unidecode
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def conversion_snake_case(data):
    data = data.rename(columns = lambda x: unidecode.unidecode(x.lower().replace(" ", "_")))
    return data


def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.03,
                head_length=0.03, 
                width=0.01, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i],
                fontsize=6)
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)

def display_factorial_planes(X_projected, 
                             x_y, 
                             pca=None, 
                             labels=None,
                             clusters=None, 
                             alpha=1,
                             figsize=[10, 8], 
                             marker=".",                              
                             show_centroids=False, 
                             centroid_marker="X", 
                             centroid_size=200,
                             x_axis_desc=None,
                             y_axis_desc=None,
                             color_palette=None):
    """
    Affiche la projection des individus sur un plan factoriel avec options d'annotation des axes.

    Paramètres supplémentaires :
    - x_axis_desc : tuple(str, str) -> (positif, négatif) pour l'axe X
    - y_axis_desc : tuple(str, str) -> (positif, négatif) pour l'axe Y
    - color_palette : liste de couleurs hexadécimales à utiliser pour les clusters
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    X_ = np.array(X_projected)

    if not len(x_y) == 2: 
        raise AttributeError("2 axes sont demandés")   
    if max(x_y) >= X_.shape[1]: 
        raise AttributeError("La variable axis n'est pas correcte")   

    x, y = x_y
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if clusters is not None:
        clusters_arr = np.array(clusters)
        unique_clusters = np.unique(clusters_arr)
        nb_clusters = len(unique_clusters)

        if color_palette is not None:
            # étendre la palette si nécessaire
            extended_palette = (color_palette * ((nb_clusters // len(color_palette)) + 1))[:nb_clusters]
        else:
            # fallback palette matplotlib si aucune palette personnalisée
            color_map = cm.get_cmap("Set1", nb_clusters)
            extended_palette = [mcolors.to_hex(color_map(i)) for i in range(nb_clusters)]

        cluster_colors = {cluster: extended_palette[i] for i, cluster in enumerate(unique_clusters)}

        for cluster in unique_clusters:
            mask = clusters_arr == cluster
            ax.scatter(X_[mask, x], X_[mask, y], 
                       alpha=alpha, 
                       c=cluster_colors[cluster], 
                       label=f'Cluster {cluster}',
                       marker=marker)

            if show_centroids:
                centroid = X_[mask][:, [x, y]].mean(axis=0)
                ax.scatter(*centroid,
                           marker=centroid_marker,
                           s=centroid_size,
                           c=cluster_colors[cluster],
                           edgecolor='black',
                           linewidth=1.5)
    else:
        ax.scatter(X_[:, x], X_[:, y], alpha=alpha, c="blue", marker=marker)

    if pca: 
        v1 = str(round(100 * pca.explained_variance_ratio_[x])) + " %"
        v2 = str(round(100 * pca.explained_variance_ratio_[y])) + " %"
    else: 
        v1 = v2 = ''

    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    x_max = np.abs(X_[:, x]).max() * 1.1
    y_max = np.abs(X_[:, y]).max() * 1.1
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom=-y_max, top=y_max)

    ax.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    ax.plot([0, 0], [-y_max, y_max], color='grey', alpha=0.8)

    offset_ratio = 1.05

    if x_axis_desc and len(x_axis_desc) == 2:
        ax.text(x_max * offset_ratio, 0, x_axis_desc[0], fontsize=12, color='black',
                ha='left', va='center')
        ax.text(-x_max * offset_ratio, 0, x_axis_desc[1], fontsize=12, color='black',
                ha='right', va='center')

    if y_axis_desc and len(y_axis_desc) == 2:
        ax.text(0, y_max * offset_ratio, y_axis_desc[0], fontsize=12, color='black',
                ha='center', va='bottom')
        ax.text(0, -y_max * offset_ratio, y_axis_desc[1], fontsize=12, color='black',
                ha='center', va='top')

    if labels is not None and len(labels): 
        for i, (_x, _y) in enumerate(X_[:, [x, y]]):
            ax.text(_x, _y + 0.05, labels[i], fontsize=12, ha='center', va='center')

    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})", pad=40)

    if clusters is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()

