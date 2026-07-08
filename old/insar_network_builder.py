import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


ROOT_DIR = r"C:\bcosta\DS-InSAR"
ALL_IFGS_CSV = os.path.join(ROOT_DIR, "interferogram_coherence_ranking.csv")
ANCHORS_CSV = os.path.join(ROOT_DIR, "highest_coh_time_series.csv")

OUT_CHAIN_CSV = os.path.join(ROOT_DIR, "highest_per_year_full_chain.csv")
OUT_PROBLEMS_TXT = os.path.join(ROOT_DIR, "highest_per_year_full_chain_problems.txt")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "network_images")
OUT_PNG = os.path.join(IMG_DIR, "full_chain_x_time_y_coherence.png")


def yyyymmdd_to_dt(s):
    return pd.to_datetime(str(s), format="%Y%m%d", errors="coerce")


def load_all_ifgs(min_coh=0.2):
    df = pd.read_csv(ALL_IFGS_CSV)
    if "Date_Pair" not in df.columns or "Mean_Coherence" not in df.columns:
        raise ValueError("interferogram_coherence_ranking.csv must contain Date_Pair and Mean_Coherence")

    tmp = df["Date_Pair"].astype(str).str.split("_", expand=True)
    df["Date1"] = tmp[0]
    df["Date2"] = tmp[1]

    df["Date1_dt"] = df["Date1"].apply(yyyymmdd_to_dt)
    df["Date2_dt"] = df["Date2"].apply(yyyymmdd_to_dt)
    df["Mean_Coherence"] = pd.to_numeric(df["Mean_Coherence"], errors="coerce")

    df = df.dropna(subset=["Date1_dt", "Date2_dt", "Mean_Coherence"]).copy()
    df = df[df["Mean_Coherence"] >= float(min_coh)].copy()
    df = df[df["Date2_dt"] > df["Date1_dt"]].copy()
    return df


def load_anchors():
    anchors = pd.read_csv(ANCHORS_CSV)
    required = {"Step", "Date1", "Date2", "Date_Pair", "Mean_Coherence"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"highest_coh_time_series.csv missing columns: {sorted(missing)}")

    anchors["Step"] = pd.to_numeric(anchors["Step"], errors="coerce")
    anchors = anchors.dropna(subset=["Step"]).copy()
    anchors["Step"] = anchors["Step"].astype(int)

    anchors["Date1_dt"] = anchors["Date1"].apply(yyyymmdd_to_dt)
    anchors["Date2_dt"] = anchors["Date2"].apply(yyyymmdd_to_dt)
    anchors = anchors.dropna(subset=["Date1_dt", "Date2_dt"]).copy()
    anchors = anchors.sort_values("Step").copy()
    return anchors


def build_within_year_graph(df_all, year):
    """Edges where BOTH endpoints are within the given calendar year."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)

    d = df_all[
        (df_all["Date1_dt"] >= start) & (df_all["Date1_dt"] < end) &
        (df_all["Date2_dt"] >= start) & (df_all["Date2_dt"] < end)
    ].copy()

    G = nx.DiGraph()
    for _, r in d.iterrows():
        u = r["Date1"]
        v = r["Date2"]
        coh = float(r["Mean_Coherence"])
        cost = 1.0 - coh  # maximize coherence -> minimize cost
        G.add_edge(u, v, coherence=coh, cost=cost, date_pair=r["Date_Pair"])
    return G


def shortest_best_path(G, src, dst):
    """Best path by coherence (min cost). Returns list of nodes or None."""
    if src not in G.nodes or dst not in G.nodes:
        return None
    try:
        return nx.shortest_path(G, source=src, target=dst, weight="cost")
    except nx.NetworkXNoPath:
        return None


def edges_from_path(G, node_path):
    rows = []
    for i in range(len(node_path) - 1):
        u = node_path[i]
        v = node_path[i + 1]
        d = G.get_edge_data(u, v) or {}
        rows.append({
            "Kind": "Connector",
            "Date1": u,
            "Date2": v,
            "Date_Pair": d.get("date_pair", f"{u}_{v}"),
            "Mean_Coherence": d.get("coherence", np.nan),
        })
    return rows


def plot_chain_segments(df_chain, out_png):
    fig, ax = plt.subplots(figsize=(18, 6))

    df = df_chain.copy()
    df["Date1_dt"] = df["Date1"].apply(yyyymmdd_to_dt)
    df["Date2_dt"] = df["Date2"].apply(yyyymmdd_to_dt)

    # Different style for anchors vs connectors
    for _, r in df.iterrows():
        if r["Kind"] == "Anchor":
            color = "black"
            lw = 3.5
            alpha = 0.9
        else:
            color = "firebrick"
            lw = 2.0
            alpha = 0.45

        ax.hlines(y=r["Mean_Coherence"], xmin=r["Date1_dt"], xmax=r["Date2_dt"],
                  colors=color, linewidth=lw, alpha=alpha)

    ax.set_title("Full chain: anchors (black) + within-year connectors (red)\nX = time, Y = mean coherence")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean coherence")
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(min_coh=0.2):
    os.makedirs(IMG_DIR, exist_ok=True)

    df_all = load_all_ifgs(min_coh=min_coh)
    anchors = load_anchors()

    chain_rows = []
    problems = []

    # Add anchors + connectors between consecutive anchors
    for i in range(len(anchors)):
        a = anchors.iloc[i]

        # 1) Always add the anchor itself (even if spanning years)
        chain_rows.append({
            "Kind": "Anchor",
            "Anchor_Step": int(a["Step"]),
            "Date1": str(a["Date1"]),
            "Date2": str(a["Date2"]),
            "Date_Pair": str(a["Date_Pair"]),
            "Mean_Coherence": float(a["Mean_Coherence"]),
        })

        # 2) Add connector from end of this anchor -> start of next anchor
        if i < len(anchors) - 1:
            next_a = anchors.iloc[i + 1]
            src = str(a["Date2"])          # end of current anchor
            dst = str(next_a["Date1"])     # start of next anchor

            src_dt = yyyymmdd_to_dt(src)
            dst_dt = yyyymmdd_to_dt(dst)

            if pd.isna(src_dt) or pd.isna(dst_dt):
                problems.append(f"Bad dates for connector: {src} -> {dst}")
                continue

            # Connector edges must be within ONE year; therefore src and dst must be in same year
            if src_dt.year != dst_dt.year:
                problems.append(
                    f"Cannot build within-year connector: {src}({src_dt.year}) -> {dst}({dst_dt.year})"
                )
                continue

            year = int(src_dt.year)
            G_year = build_within_year_graph(df_all, year)

            path_nodes = shortest_best_path(G_year, src, dst)
            if path_nodes is None:
                problems.append(f"No within-year path in {year} for {src} -> {dst}")
                continue

            connector_edges = edges_from_path(G_year, path_nodes)
            for j, r in enumerate(connector_edges, start=1):
                r.update({
                    "Anchor_Step": int(a["Step"]),
                    "Connector_To_Anchor_Step": int(next_a["Step"]),
                    "Connector_Year": year,
                    "Connector_Index": j,
                })
                chain_rows.append(r)

    df_chain = pd.DataFrame(chain_rows)
    df_chain.to_csv(OUT_CHAIN_CSV, index=False)
    print(f"Saved chain CSV: {OUT_CHAIN_CSV}")

    plot_chain_segments(df_chain, OUT_PNG)
    print(f"Saved plot: {OUT_PNG}")

    if problems:
        with open(OUT_PROBLEMS_TXT, "w") as f:
            f.write("\n".join(problems))
        print(f"Some connectors could not be built. See: {OUT_PROBLEMS_TXT}")
    else:
        print("All connectors built successfully.")


if __name__ == "__main__":
    main(min_coh=0.2)
