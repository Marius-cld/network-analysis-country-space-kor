"""Analyse de réseaux du Country Space (similarité des paniers d'exportations
entre pays), 2008 vs 1998 : statistiques descriptives, distributions du
degré/force, centralités, indice petit-monde (Neal, 2017), comparaison
2008/1998 par régression MRQAP, robustesse (valeur de Fiedler, arbre couvrant
minimal), et position de la Corée du Sud dans le réseau.

Doit être lancé avec `data/`, `figure/` et `table/` accessibles à la racine
du dépôt (chemins résolus depuis l'emplacement de ce script).
"""

from pathlib import Path

import community.community_louvain as louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm

from libs.mrqap import MRQAP

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURE_DIR = ROOT_DIR / "figure"
TABLE_DIR = ROOT_DIR / "table"


### Chargement des données ###

def charger_matrices():
    df = pd.read_csv(DATA_DIR / "Country_Space_2008.csv")
    df = df.set_index("iso3").drop(columns=["country_code_BACI", "country_name"])

    df_1 = pd.read_csv(DATA_DIR / "Country_Space_1998.csv")
    df_1 = df_1.set_index("iso3").drop(columns=["country_code_BACI", "country_name"])

    for nom, matrice in (("2008", df), ("1998", df_1)):
        diagonale = np.diag(matrice.to_numpy())
        print(f"[{nom}] carrée : {matrice.shape[0] == matrice.shape[1]} | "
              f"diagonale nulle : {(diagonale == 0).all()} | "
              f"symétrique : {np.allclose(matrice, matrice.T)} | "
              f"coefficients positifs : {np.all(matrice >= 0)}")

    return df, df_1


### I - Analyse du réseau mondial ###

def dessiner_reseau(graph, filename, titre, pondere=False, mst=None):
    plt.figure(figsize=(16, 12), dpi=200)
    node_colors = ["#0666CC" if node == "KOR" else "#4DA6FF" for node in graph.nodes()]
    pos = nx.kamada_kawai_layout(graph)
    if "TGO" in pos:
        pos["TGO"] = (0.3, -0.3)

    if pondere:
        d = dict(graph.degree)
        node_sizes = [v * 3 for v in d.values()]
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="#4DA6FF", linewidths=1)
        for node, degree in d.items():
            taille_finale = max(degree * 0.05, 6)
            nx.draw_networkx_labels(
                graph, pos, labels={node: node}, font_size=taille_finale,
                font_color="#003366", font_weight="bold" if degree > 100 else "normal",
            )
        nx.draw_networkx_edges(graph, pos, edge_color="#AFCFF7", alpha=0.7, width=0.5)
    else:
        nx.draw_networkx_edges(graph, pos, alpha=0.7, edge_color="#AFCFF7")
        if mst is not None:
            nx.draw_networkx_edges(mst, pos, edge_color="#e68481", width=2)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_labels(graph, pos, font_size=8)

    plt.title(titre, fontsize=20)
    plt.axis("off")
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()


def decrire_reseau(graph, rapport, titre):
    """Notions : diamètre = plus grande distance entre deux sommets, rayon =
    plus petite excentricité, centre = sommets d'excentricité minimale,
    périphérie = sommets d'excentricité maximale, barycentre = sommets
    minimisant la distance moyenne aux autres."""
    rapport.append(f"\n=== {titre} ===")

    nb_node = graph.number_of_nodes()
    nb_edge = graph.number_of_edges()
    connexe = nx.is_connected(graph)
    rapport.append(f"Nombre de noeuds : {nb_node}")
    rapport.append(f"Nombre d'arêtes : {nb_edge}")
    rapport.append(f"Orienté : {graph.is_directed()}")
    rapport.append(f"Connexe : {connexe}")
    rapport.append(f"Nombre de composantes connexes : {nx.number_connected_components(graph)}")
    rapport.append(f"Densité : {nx.density(graph)}")

    if connexe:
        cible = graph
        label = ""
    else:
        rapport.append("Graphe non connexe : calculs sur la plus grande composante")
        composantes = list(nx.connected_components(graph))
        cible = graph.subgraph(max(composantes, key=len)).copy()
        label = " (composante géante)"

    rapport.append(f"Diamètre{label} : {nx.diameter(cible)}")
    rapport.append(f"Rayon{label} : {nx.radius(cible)}")
    rapport.append(f"Centre{label} : {nx.center(cible)}")
    rapport.append(f"Sommets périphériques{label} : {nx.periphery(cible)}")
    rapport.append(f"Barycentre{label} : {nx.center(cible, usebounds=True)}")
    rapport.append(f"Barycentre valué{label} : {nx.barycenter(cible, weight='weight')}")

    # Degré/force. Degré = nombre de liens par noeud, force = somme des
    # poids des liens par noeud (degré pondéré).
    degres = dict(graph.degree())
    rapport.append(f"Degré moyen : {sum(degres.values()) / nb_node}")
    rapport.append(f"Degré maximum : {max(degres, key=degres.get)} ({max(degres.values())})")
    rapport.append(f"Degré minimum : {min(degres, key=degres.get)} ({min(degres.values())})")

    forces = dict(graph.degree(weight="weight"))
    rapport.append(f"Force moyenne : {sum(forces.values()) / nb_node}")
    rapport.append(f"Force maximum : {max(forces, key=forces.get)} ({max(forces.values())})")
    rapport.append(f"Force minimum : {min(forces, key=forces.get)} ({min(forces.values())})")

    # Centralisation globale du réseau : concentration autour de quelques
    # noeuds (proche de 1) vs réseau dispersé (proche de 0).
    hub_max = max(degres.values())
    centralisation = sum(hub_max - v for v in degres.values()) / ((nb_node - 1) * (nb_node - 2))
    rapport.append(f"Centralisation du graphe : {centralisation}")

    return degres, forces


def graphique_distribution(sequence, xlabel, filename_prefix, bin_step):
    # Histogramme
    plt.figure(dpi=200)
    bins = np.arange(min(sequence), max(sequence) + bin_step, bin_step)
    plt.hist(sequence, bins=bins, edgecolor="black", align="left")
    plt.title(f"Distribution {xlabel.lower()} du graphe")
    plt.xlabel(xlabel)
    plt.ylabel("Fréquence")
    plt.savefig(FIGURE_DIR / f"{filename_prefix}_hist.png")
    plt.close()

    sequence_triee = sorted(sequence, reverse=True)

    # Classement (courbe ligne)
    plt.figure(dpi=200)
    plt.plot(sequence_triee, "r-", marker="o", linewidth=1, markersize=2)
    plt.title(f"Classement des {xlabel.lower()}s")
    plt.xlabel("Rang")
    plt.ylabel(xlabel)
    plt.savefig(FIGURE_DIR / f"{filename_prefix}_ligne.png")
    plt.close()

    # Échelle log-log
    plt.figure(dpi=200)
    plt.loglog(sequence_triee, "go-", linewidth=1, markersize=2)
    plt.title(f"Distribution {xlabel.lower()} (log-log)")
    plt.xlabel(xlabel)
    plt.ylabel("Fréquence")
    plt.savefig(FIGURE_DIR / f"log_{filename_prefix}.png")
    plt.close()


def acteurs_centraux(graph, rapport):
    """Degré, proximité (closeness), intermédiarité (betweenness), autonomie
    (eigenvector), charge (load) : cinq mesures de centralité classiques."""
    mesures = {
        "Centralité de degré (degree)": nx.degree_centrality(graph),
        "Centralité de proximité (closeness)": nx.closeness_centrality(graph),
        "Centralité d'intermédiarité (betweenness)": nx.betweenness_centrality(graph, normalized=True),
        "Centralité d'autonomie (eigenvector)": nx.eigenvector_centrality(graph, max_iter=1000),
        "Centralité de charge (load)": nx.load_centrality(graph),
    }

    rapport.append("\n=== Acteurs centraux (top 5) ===")
    tables = {}
    for nom, valeurs in mesures.items():
        classement = sorted(valeurs.items(), key=lambda item: item[1], reverse=True)
        rapport.append(f"{nom} : {classement[:5]}")
        colonne = nom.split("(")[-1].rstrip(")")
        tables[colonne] = pd.DataFrame(classement, columns=["iso3", nom]).sort_values(by=nom, ascending=False)

    df_acteur = tables["degree"]
    for colonne in ("closeness", "betweenness", "eigenvector", "load"):
        df_acteur = df_acteur.merge(tables[colonne], on="iso3")

    with pd.ExcelWriter(TABLE_DIR / "centralites-tous-acteurs.xlsx") as writer:
        df_acteur.to_excel(writer, sheet_name="Ensemble", index=False)
        for i, (colonne, table) in enumerate(tables.items(), start=1):
            table.head(5).to_excel(writer, sheet_name=f"Top5_{colonne}", index=False)

    return mesures, df_acteur


def acteurs_peripheriques(mesures, df_acteur, rapport):
    """Les acteurs périphériques sont ceux dont la centralité se démarque
    par des valeurs faibles (seuils fixés empiriquement à partir de la
    distribution de chaque mesure)."""
    seuils = {
        "Centralité de charge (load)": 0.0002,
        "Centralité de degré (degree)": 0.03,
        "Centralité de proximité (closeness)": 0.35,
        "Centralité d'intermédiarité (betweenness)": 0.0001,
        "Centralité d'autonomie (eigenvector)": 0.0001,
    }

    rapport.append("\n=== Acteurs périphériques (centralité < seuil) ===")
    tables_filtrees = {}
    for nom, seuil in seuils.items():
        colonne = nom.split("(")[-1].rstrip(")")
        table = pd.DataFrame(mesures[nom].items(), columns=["iso3", "centralite"]).sort_values("centralite")
        table_filtree = table.loc[table["centralite"] < seuil]
        rapport.append(f"{nom} (< {seuil}) : {len(table_filtree)} pays")
        tables_filtrees[colonne] = table_filtree

    with pd.ExcelWriter(TABLE_DIR / "centralites-acteurs-peripheriques.xlsx") as writer:
        df_acteur.to_excel(writer, sheet_name="Ensemble", index=False)
        for i, (colonne, table) in enumerate(tables_filtrees.items(), start=1):
            table.to_excel(writer, sheet_name=f"Peripherique_{colonne}", index=False)


def indice_petit_monde(graph, rapport):
    """Small-World Index (Neal, 2017) : compare la composante géante à un
    graphe aléatoire (Erdos-Renyi) et un graphe en grille (Watts-Strogatz,
    p=0) de mêmes taille/densité/degré moyen."""
    largest_cc = max(nx.connected_components(graph), key=len)
    g = graph.subgraph(largest_cc).copy()
    n = g.number_of_nodes()
    density = nx.density(g)
    k = int(round(sum(dict(g.degree()).values()) / n))

    l_obs = nx.average_shortest_path_length(g)
    c_obs = nx.average_clustering(g)

    g_r = nx.erdos_renyi_graph(n=n, p=density)
    l_r = nx.average_shortest_path_length(g_r)
    c_r = nx.average_clustering(g_r)

    g_l = nx.watts_strogatz_graph(n=n, k=k, p=0)
    l_l = nx.average_shortest_path_length(g_l)
    c_l = nx.average_clustering(g_l)

    swi = ((l_obs - l_l) / (l_r - l_l)) * ((c_obs - c_r) / (c_l - c_r))

    rapport.append("\n=== Réseau petit-monde (Small-World Index, Neal 2017) ===")
    rapport.append(f"Small-World Index : {swi}")
    rapport.append(f"Coefficient de clustering : {c_obs}")
    rapport.append(f"Distance moyenne entre les noeuds : {l_obs}")


### Comparaison 2008 / 1998 : MRQAP ###

def vectorize(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]


def mrqap_manuel(df, df_1, rapport, n_permutations=1000):
    common_nodes = df.index.intersection(df_1.index)
    a = df.loc[common_nodes, common_nodes].to_numpy()
    b = df_1.loc[common_nodes, common_nodes].to_numpy()

    corr = np.corrcoef(vectorize(a), vectorize(b))[0, 1]
    rapport.append(f"\n=== MRQAP (approche manuelle par permutations) ===")
    rapport.append(f"Dimensions des matrices alignées : {a.shape}")
    rapport.append(f"Corrélation 2008 / 1998 : {corr:.4f}")

    y_vec = vectorize(a)
    x_vec = vectorize(b).reshape(-1, 1)
    x_vec_const = sm.add_constant(x_vec)

    model = sm.OLS(y_vec, x_vec_const).fit()

    rng = np.random.default_rng(42)
    coefs_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(a.shape[0])
        a_perm = a[perm, :][:, perm]
        coefs_perm[i] = sm.OLS(vectorize(a_perm), x_vec_const).fit().params[1]

    coef_obs = model.params[1]
    p_value = (np.abs(coefs_perm) >= np.abs(coef_obs)).mean()

    with open(TABLE_DIR / "mrqap-manuel.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f"\n\nCoefficient observé : {coef_obs}\n")
        f.write(f"P-value MRQAP (permutations) : {p_value}\n")

    rapport.append(f"Coefficient observé : {coef_obs} | P-value (permutations) : {p_value}")


def mrqap_bibliotheque(df, df_1, rapport, n_permutations=1000):
    common_nodes = df.index.intersection(df_1.index)
    a = df.loc[common_nodes, common_nodes].to_numpy()
    b = df_1.loc[common_nodes, common_nodes].to_numpy()

    mrqap = MRQAP(Y={"Reseau_2008": a}, X={"Reseau_1998": b}, npermutations=n_permutations, diagonal=False, directed=True)
    mrqap.mrqap()

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        mrqap.summary()
    summary_text = buffer.getvalue()

    with open(TABLE_DIR / "mrqap-bibliotheque.tex", "w") as f:
        f.write("\\begin{verbatim}\n")
        f.write(summary_text)
        f.write("\\end{verbatim}")

    rapport.append("\n=== MRQAP (bibliothèque mrqap-python) ===")
    rapport.append(summary_text)
    rapport.append(
        "Interprétation : le coefficient de x1 indique qu'une augmentation d'une unité de x1 "
        "(réseau 1998) entraîne en moyenne une hausse de sa valeur de y (réseau 2008) ; un R² "
        "faible indique que la structure de 1998 n'explique qu'une part marginale de celle de "
        "2008, signe d'une recomposition du commerce mondial sur la période."
    )


### Robustesse : valeur de Fiedler ###

def valeur_fiedler(graph, rapport, label):
    """La valeur de Fiedler (λ2, deuxième plus petite valeur propre de la
    matrice laplacienne) quantifie la difficulté à déconnecter le réseau."""
    degree_weighted = dict(graph.degree(weight="weight"))
    a = nx.adjacency_matrix(graph, weight="weight").todense()
    d = np.diag([degree_weighted[n] for n in graph.nodes()])
    laplacian = d - a

    eigvals = np.sort(np.real(np.linalg.eigvals(laplacian)))
    fiedler_value = nx.algebraic_connectivity(graph, weight="weight")

    rapport.append(f"\n=== Robustesse ({label}) ===")
    rapport.append(f"5 plus petites valeurs propres : {eigvals[:5]}")
    rapport.append(f"λ2 (calcul manuel) : {eigvals[1]}")
    rapport.append(f"Valeur de Fiedler (nx.algebraic_connectivity) : {fiedler_value}")


### II - Position de la Corée du Sud ###

def reseau_coree(graph):
    ego_kor = nx.ego_graph(graph, "KOR")

    plt.figure(figsize=(16, 12), dpi=200)
    node_colors = ["#0666CC" if node == "KOR" else "#4DA6FF" for node in ego_kor.nodes()]
    pos = nx.kamada_kawai_layout(ego_kor)
    nx.draw_networkx_edges(ego_kor, pos, alpha=0.7, edge_color="#AFCFF7")
    nx.draw_networkx_nodes(ego_kor, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(ego_kor, pos, font_size=9)
    plt.title("Réseau de la Corée du Sud (2008)", fontsize=20)
    plt.axis("off")
    plt.savefig(FIGURE_DIR / "reseau_coree_2008.png", bbox_inches="tight")
    plt.close()


def coree_acteur_central(graph, rapport):
    mesures = {
        "Centralité de charge": nx.load_centrality(graph),
        "Centralité de degré": nx.degree_centrality(graph),
        "Centralité de proximité": nx.closeness_centrality(graph),
        "Centralité d'intermédiarité": nx.betweenness_centrality(graph, normalized=True),
        "Centralité d'autonomie": nx.eigenvector_centrality(graph, max_iter=1000),
    }

    rapport.append("\n=== La Corée est-elle un acteur central ? ===")
    df_centralites = pd.DataFrame({nom: valeurs for nom, valeurs in mesures.items()})
    for nom, valeurs in mesures.items():
        premier = max(valeurs.items(), key=lambda x: x[1])
        rapport.append(f"{nom} — premier pays : {premier} | Corée (KOR) : {valeurs.get('KOR')}")

    # Percentile : plus la valeur est basse, plus le pays est central.
    df_top_percent = (1 - df_centralites.rank(pct=True)) * 100
    rapport.append(f"Rang percentile de la Corée : {df_top_percent.loc['KOR'].to_dict()}")

    return mesures["Centralité de degré"]


def voisins_coree(graph):
    neighbors = graph["KOR"]
    # On regarde les 15 premiers voisins de la Corée pour déterminer le top
    # le plus pertinent.
    top15 = sorted(neighbors.items(), key=lambda x: x[1]["weight"], reverse=True)[:15]
    df_top15 = pd.DataFrame([{"ISO3": country, "weight": attrs["weight"]} for country, attrs in top15])
    df_top15.to_excel(TABLE_DIR / "korea-voisins-top15.xlsx", index=False)
    return top15


def cluster_coree(graph, rapport):
    partition = louvain.best_partition(graph, random_state=42)
    df_partition = (
        pd.DataFrame.from_dict(partition, orient="index", columns=["community"])
        .reset_index()
        .rename(columns={"index": "iso3"})
        .sort_values("community")
        .reset_index(drop=True)
    )
    df_partition.to_excel(TABLE_DIR / "louvain-clusters.xlsx", index=False)

    kor_community = df_partition.loc[df_partition["iso3"] == "KOR", "community"].values[0]
    kor_cluster = df_partition[df_partition["community"] == kor_community]

    rapport.append(f"\n=== Cluster (Louvain) de la Corée ===")
    rapport.append(f"KOR appartient à la communauté {kor_community} ({len(kor_cluster)} pays)")

    kor_cluster_nodes = kor_cluster["iso3"].tolist()
    subgraph = graph.subgraph(kor_cluster_nodes)

    plt.figure(figsize=(16, 12), dpi=200)
    node_colors = [
        "#FF6B35" if node == "KOR" else "#E63946" if node == "USA" else "#4DA6FF"
        for node in subgraph.nodes()
    ]
    pos = nx.kamada_kawai_layout(subgraph)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.7, edge_color="#AFCFF7")
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.title(f"Cluster {kor_community} contenant KOR", fontsize=20)
    plt.axis("off")
    plt.savefig(FIGURE_DIR / "cluster_kor.png", bbox_inches="tight")
    plt.close()


def distance_coree_usa(graph_2008, rapport):
    graph_dist = graph_2008.copy()
    # Poids de similarité -> distance (similarité élevée = distance faible),
    # pour que Dijkstra privilégie les liens les plus similaires.
    for _, _, d in graph_dist.edges(data=True):
        d["weight"] = 1 - d["weight"]

    chemin = nx.dijkstra_path(graph_dist, "KOR", "USA")
    longueur = nx.dijkstra_path_length(graph_dist, "KOR", "USA")

    rapport.append("\n=== Distance Corée du Sud - États-Unis ===")
    rapport.append(f"Plus court chemin : {chemin}")
    rapport.append(f"Longueur cumulée : {longueur}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rapport = []

    df, df_1 = charger_matrices()

    ### I - Réseau mondial (2008) ###
    graph = nx.from_pandas_adjacency(df)

    dessiner_reseau(graph, "reseau_monde_2008_non_pondere.png", "Réseau Monde (2008)")
    dessiner_reseau(graph, "reseau_monde_2008.png", "Réseau Monde Pondéré (2008)", pondere=True)

    degres, forces = decrire_reseau(graph, rapport, "Réseau Monde (2008)")
    graphique_distribution(list(degres.values()), "Degré", "degree", bin_step=10)
    graphique_distribution(list(forces.values()), "Force", "force", bin_step=3)

    mesures, df_acteur = acteurs_centraux(graph, rapport)
    acteurs_peripheriques(mesures, df_acteur, rapport)
    indice_petit_monde(graph, rapport)

    ### Comparaison avec 1998 ###
    graph_1998 = nx.from_pandas_adjacency(df_1)
    dessiner_reseau(graph_1998, "reseau_monde_1998_non_pondere.png", "Réseau Monde (1998)")
    dessiner_reseau(graph_1998, "reseau_monde_1998.png", "Réseau Monde Pondéré (1998)", pondere=True)

    mrqap_manuel(df, df_1, rapport)
    mrqap_bibliotheque(df, df_1, rapport)

    ### Robustesse ###
    valeur_fiedler(graph, rapport, "graphe complet")
    largest_cc = max(nx.connected_components(graph), key=len)
    valeur_fiedler(graph.subgraph(largest_cc).copy(), rapport, "composante géante")

    mst = nx.minimum_spanning_tree(graph)
    rapport.append(f"\n=== Arbre couvrant minimal ===")
    rapport.append(f"Poids total du MST : {mst.size(weight='weight')}")
    dessiner_reseau(graph, "reseau_monde_2008_mst_overlay.png", "Réseau Monde (2008)", mst=mst)
    dessiner_reseau(mst, "spanning_tree.png", "MST Réseau Monde (2008)")

    ### II - Position de la Corée du Sud ###
    reseau_coree(graph)
    coree_acteur_central(graph, rapport)
    voisins_coree(graph)
    cluster_coree(graph, rapport)
    distance_coree_usa(graph, rapport)

    with open(TABLE_DIR / "descriptif-reseau-2008.txt", "w") as f:
        f.write("\n".join(str(ligne) for ligne in rapport))

    print("Terminé.")


if __name__ == "__main__":
    main()
