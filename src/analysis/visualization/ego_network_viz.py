from __future__ import annotations

import os
from pathlib import Path

from ...config import EGO_COLOR, ALTER_COLOR
from ...auxiliary import short_name


# ─────────────────────────────────────────────
# EGO-NETWORK VISUALISATION
# ─────────────────────────────────────────────

def visualise_ego(
    ego: "nx.Graph | nx.DiGraph",
    ego_id: int,
    name: str,
    out_dir: Path,
) -> None:
    """Save an interactive pyvis HTML for a single ego-network.

    The ego is pinned at the centre; alters are arranged in a circle around it.
    """
    import math
    from pyvis.network import Network

    net = Network(
        notebook=False, height="700px", width="100%",
        directed=ego.is_directed(),
    )
   
    # Use repulsion so alter-alter edges still look tidy, but keep
    # central_gravity at 0 so the pinned ego stays put.
    net.repulsion(
        node_distance=180, central_gravity=0.0,
        spring_length=200, spring_strength=0.05, damping=0.09,
    )

    # Separate ego from alters
    alter_nodes = [n for n in ego.nodes() if n != ego_id]
    n_alters = len(alter_nodes)
    radius = 300  # pixel radius of the orbit

    # --- Ego: pinned at (0, 0) ---
    ego_attrs = ego.nodes[ego_id] if ego_id in ego else {}
    net.add_node(
        ego_id,
        label=short_name(ego_attrs.get("name", str(ego_id))),
        color=EGO_COLOR,
        size=35,
        title=ego_attrs.get("name", str(ego_id)),
        x=0, y=0, fixed=True,
        physics=False,
    )

    # --- Alters: evenly spaced on a circle ---
    for i, node in enumerate(alter_nodes):
        attrs = ego.nodes[node]
        angle = 2 * math.pi * i / max(n_alters, 1)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        net.add_node(
            node,
            label=short_name(attrs.get("name", str(node))),
            color=ALTER_COLOR,
            size=18,
            title=attrs.get("name", str(node)),
            x=x, y=y,
        )

    for u, v, attrs in ego.edges(data=True):
        w = attrs.get("weight", 1)
        net.add_edge(u, v, value=w, title=f"weight: {w}")

    safe = short_name(name).replace(" ", "_")
    net.save_graph(str(out_dir / f"ego_{safe}.html"))


def visualise_shared_alters(
    G: "nx.Graph | nx.DiGraph",
    shared_ids: set[int],
    victim_ids: set[int],
    out_dir: Path,
) -> None:
    """
    Build a subgraph of shared alters + victims and save as interactive HTML.
    """
    from pyvis.network import Network
    keep = shared_ids | victim_ids
    sub = G.subgraph(keep).copy()

    if sub.number_of_nodes() == 0:
        return

    net = Network(
        notebook=False, height="800px", width="100%",
        directed=sub.is_directed(),
    )
    net.repulsion(
        node_distance=200, central_gravity=0.2,
        spring_length=250, spring_strength=0.05, damping=0.09,
    )
    print("PASSED")

    for node, attrs in sub.nodes(data=True):
        label = short_name(attrs.get("name", str(node)))
        if node in victim_ids:
            color = EGO_COLOR
            size = 25
        else:
            color = ALTER_COLOR
            size = 20

        print(f'{node=}, {type(node)=}')
        print(f'{label=}, {color=}, {size=}, {attrs.get("name", str(node))=}')
        net.add_node(int(node), label=label, color=color, size=size,
                     title=attrs.get("name", str(node)))

    for u, v, attrs in sub.edges(data=True):
        w = attrs.get("weight", 1)
        net.add_edge(u, v, value=w, title=f"weight: {w}")

    net.save_graph(str(out_dir / "shared_alters.html"))


def visualise_heatmap(
    overlap_mat: "pd.DataFrame", out_dir: Path
) -> None:
    """Save a pairwise-overlap heatmap as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(overlap_mat.values, cmap="YlOrRd", aspect="auto")

    labels = [short_name(c) for c in overlap_mat.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(overlap_mat.values[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if overlap_mat.values[i, j] > overlap_mat.values.max() / 2 else "black")

    plt.colorbar(im, ax=ax, label="Shared Alters")
    ax.set_title("Pairwise Alter Overlap Between Victims")
    plt.tight_layout()
    plt.savefig(out_dir / "overlap_heatmap.png", dpi=150)
    plt.close()
