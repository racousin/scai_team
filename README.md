# scai_team — How does a model see the world?

A ~5-minute workshop notebook: short text bios (the SCAI team, cartoon characters, politicians) are embedded by **two small Hugging Face models** and displayed in 2D embedding space — two different world views of the same texts. Run it, look, then edit the texts and watch the maps change.

**▶️ Open the notebook in Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/scai_team/blob/main/how_models_see_the_world.ipynb)

## Contents

- [`how_models_see_the_world.ipynb`](how_models_see_the_world.ipynb) — the notebook: the texts, the two model names, and a couple of function calls. That's all.
- [`embedding_utils.py`](embedding_utils.py) — the plumbing (embedding, 2D projection, interactive plots). The notebook fetches this file at runtime from this repo, so it stays out of the participants' way.
