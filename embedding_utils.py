"""Helpers for the "How does a model see the world?" workshop notebook.

The notebook only provides texts and model names; everything reusable lives here.

Expected data shape: a list of items, each item a dict with exactly these keys:
    {"name": "Homer Simpson", "group": "cartoon", "text": "Lazy but lovable ..."}
"""

import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

__all__ = [
    "embed_texts",
    "project_2d",
    "compare_models",
    "plot_embedding_space",
    "similarity_heatmap",
]

# --- chart style (validated palette; groups get hues in fixed order) ---------
_GROUP_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_HEATMAP_RAMP = [(0.0, "#cde2fb"), (0.25, "#9ec5f4"), (0.5, "#5598e7"), (0.75, "#256abf"), (1.0, "#0d366b")]
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_MODEL_CACHE = {}


def _load_model(model_name):
    if model_name not in _MODEL_CACHE:
        print(f"Loading {model_name} … (first time only)")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _validate_items(items):
    if not items:
        raise ValueError("items is empty — add at least one {'name', 'group', 'text'} entry")
    for i, item in enumerate(items):
        missing = {"name", "group", "text"} - set(item)
        if missing:
            raise ValueError(f"items[{i}] is missing key(s) {sorted(missing)}: {item}")
    names = [item["name"] for item in items]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate names {duplicates} — make each name unique, e.g. 'Macron (EN)' / 'Macron (FR)'")


def _group_colors(items):
    groups = list(dict.fromkeys(item["group"] for item in items))
    if len(groups) > len(_GROUP_COLORS):
        raise ValueError(f"{len(groups)} groups but only {len(_GROUP_COLORS)} colors — merge some groups")
    return dict(zip(groups, _GROUP_COLORS))


def embed_texts(model_name, texts):
    """Turn texts into one unit-length vector per text — the model's view of each text."""
    return _load_model(model_name).encode(list(texts), normalize_embeddings=True)


def project_2d(embeddings, method="pca"):
    """Squash high-dimensional vectors down to 2D so we can look at them."""
    if method == "pca":
        return PCA(n_components=2).fit_transform(embeddings)
    if method == "tsne":
        perplexity = min(30.0, max(2.0, (len(embeddings) - 1) / 3))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(embeddings)
    raise ValueError(f"unknown method {method!r} — use 'pca' or 'tsne'")


def _hover_bio(text):
    return "<br>".join(textwrap.wrap(text, width=55))


def _label_positions(xy):
    """Put each label on the side facing away from its nearest neighbour, so close points don't stack labels."""
    span = xy.max(axis=0) - xy.min(axis=0)
    norm = (xy - xy.min(axis=0)) / np.where(span == 0, 1, span)
    positions = []
    for i in range(len(norm)):
        distances = np.linalg.norm(norm - norm[i], axis=1)
        distances[i] = np.inf
        nearest = int(np.argmin(distances))
        positions.append("bottom center" if norm[nearest, 1] > norm[i, 1] else "top center")
    return positions


def _scatter_traces(items, xy, colors, showlegend):
    positions = _label_positions(xy)
    traces = []
    for group, color in colors.items():
        idx = [i for i, item in enumerate(items) if item["group"] == group]
        traces.append(
            go.Scatter(
                x=xy[idx, 0],
                y=xy[idx, 1],
                mode="markers+text",
                name=group,
                legendgroup=group,
                showlegend=showlegend,
                text=[items[i]["name"] for i in idx],
                textposition=[positions[i] for i in idx],
                textfont=dict(color=_INK, size=11),
                cliponaxis=False,
                marker=dict(color=color, size=11, line=dict(color=_SURFACE, width=2)),
                customdata=[_hover_bio(items[i]["text"]) for i in idx],
                hovertemplate="<b>%{text}</b><br><br>%{customdata}<extra>" + group + "</extra>",
            )
        )
    return traces


def _style_scatter_layout(fig, title, height):
    fig.update_layout(
        title=dict(text=title, font=dict(color=_INK, size=16, family=_FONT)),
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font=dict(family=_FONT, color=_MUTED, size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5, font=dict(color=_INK)),
        margin=dict(l=60, r=60, t=90, b=60),
        height=height,
        hoverlabel=dict(bgcolor=_SURFACE, font=dict(family=_FONT, color=_INK)),
    )
    # The axes of a 2D projection carry no meaning — only distances do. Hide the numbers.
    fig.update_xaxes(showticklabels=False, gridcolor=_GRID, zeroline=False, showline=False, title=None)
    fig.update_yaxes(showticklabels=False, gridcolor=_GRID, zeroline=False, showline=False, title=None)


def plot_embedding_space(items, model_name, method="pca"):
    """One model's world map: embed every item's text, project to 2D, plot it."""
    _validate_items(items)
    colors = _group_colors(items)
    xy = project_2d(embed_texts(model_name, [item["text"] for item in items]), method=method)
    fig = go.Figure(_scatter_traces(items, xy, colors, showlegend=True))
    _style_scatter_layout(fig, f"How {model_name.split('/')[-1]} sees these texts ({method.upper()} projection)", height=560)
    return fig


def compare_models(items, model_a, model_b, method="pca"):
    """Side-by-side world maps of the same texts through two different models' eyes."""
    _validate_items(items)
    colors = _group_colors(items)
    texts = [item["text"] for item in items]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(model_a.split("/")[-1], model_b.split("/")[-1]),
        horizontal_spacing=0.06,
    )
    for col, model_name in enumerate((model_a, model_b), start=1):
        xy = project_2d(embed_texts(model_name, texts), method=method)
        for trace in _scatter_traces(items, xy, colors, showlegend=(col == 1)):
            fig.add_trace(trace, row=1, col=col)
    _style_scatter_layout(fig, f"Same texts, two world views ({method.upper()} projection — only distances matter)", height=560)
    for annotation in fig.layout.annotations:  # subplot titles
        annotation.font = dict(color=_INK, size=13, family=_FONT)
    return fig


def similarity_heatmap(items, model_name):
    """Who is closest to whom? Cosine similarity between every pair of texts (1 = identical view)."""
    _validate_items(items)
    embeddings = embed_texts(model_name, [item["text"] for item in items])
    similarity = embeddings @ embeddings.T
    names = [item["name"] for item in items]
    fig = go.Figure(
        go.Heatmap(
            z=similarity,
            x=names,
            y=names,
            colorscale=_HEATMAP_RAMP,
            zmin=float(np.floor(similarity.min() * 10) / 10),
            zmax=1.0,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>similarity: %{z:.2f}<extra></extra>",
            colorbar=dict(title="cosine sim.", outlinewidth=0, tickfont=dict(color=_MUTED)),
        )
    )
    fig.update_layout(
        title=dict(text=f"Who is closest to whom, according to {model_name.split('/')[-1]}", font=dict(color=_INK, size=16, family=_FONT)),
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font=dict(family=_FONT, color=_MUTED, size=11),
        margin=dict(l=40, r=40, t=90, b=40),
        height=140 + 32 * len(items),
        hoverlabel=dict(bgcolor=_SURFACE, font=dict(family=_FONT, color=_INK)),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=_INK))
    fig.update_xaxes(tickfont=dict(color=_INK))
    return fig
