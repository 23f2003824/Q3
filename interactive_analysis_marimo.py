# Marimo interactive notebook
# Author contact: 23f2003824@ds.study.iitm.ac.in

import marimo as mo

app = mo.App()

# --- Cell 1: Imports & configuration ---
# Exposes np and pd to downstream cells.
@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd

# --- Cell 2: UI controls (widgets) ---
# Provides interactive controls used by downstream cells.
@app.cell
def _(mo):
    # Slider to control sample size
    n_slider = mo.ui.slider(start=50, stop=1000, step=50, value=300, label="Sample size (n)")

    # Slider to control noise level in y
    noise_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=1.0, label="Noise (σ)")

    # Display the controls stacked vertically
    mo.vstack([n_slider, noise_slider])
    return n_slider, noise_slider

# --- Cell 3: Synthetic data generation ---
# Depends on: np, pd, n_slider, noise_slider
# Produces: DataFrame df used by downstream analysis/plot cells.
@app.cell
def _(np, pd, n_slider, noise_slider):
    rng = np.random.default_rng(42)
    n = int(n_slider.value)
    sigma = float(noise_slider.value)

    x = rng.normal(0, 1, size=n)
    # Nonlinear relationship with noise
    y = 2 * x + 0.8 * x**2 + rng.normal(0, sigma, size=n)
    z = -x + rng.normal(0, sigma/2 + 1e-9, size=n)  # add small jitter to avoid perfect collinearity

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    return df, n, sigma

# --- Cell 4: Summary statistics and correlation ---
# Depends on: df
# Produces: corr matrix and descriptive stats.
@app.cell
def _(df):
    desc = df.describe()
    corr = df.corr(numeric_only=True)
    return corr, desc

# --- Cell 5: Dynamic markdown summary ---
# Depends on: n, sigma, corr
# Renders a reactive, human-readable explanation tied to slider state.
@app.cell
def _(mo, n, sigma, corr):
    import numpy as np
    strongest = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))  # ignore self-correlation
        .stack()
        .abs()
        .sort_values(ascending=False)
        .index[0]
    )
    mo.md(
        f"""
### Interactive summary
- **Sample size (n):** {n}  
- **Noise (σ):** {sigma:.2f}  
- **Strongest absolute correlation:** `{strongest[0]} ↔ {strongest[1]}` (|r| = {corr.loc[strongest]:.3f})

Adjust the sliders above to regenerate the data and update this summary.
"""
    )

# --- Cell 6: Show tables ---
# Depends on: desc, corr
# Displays the outputs that update when sliders change.
@app.cell
def _(mo, desc, corr):
    mo.md("#### Descriptive statistics")
    desc
    mo.md("#### Correlation matrix")
    corr

# --- Cell 7: Quick plot ---
# Depends on: df
# Simple scatter plot that reacts to widget state.
@app.cell
def _(mo, df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scatter: y vs x (reactive)")
    fig

# --- Cell 8: Documentation of data flow ---
# Pure markdown cell explaining dependencies in plain language.
@app.cell
def _(mo):
    mo.md(
        """
### Data flow documentation
1. **Cell 1** imports `numpy` and `pandas` → exports `np`, `pd`.\
2. **Cell 2** creates sliders `n_slider` and `noise_slider`.\
3. **Cell 3** **depends on** (`np`, `pd`, `n_slider`, `noise_slider`) → emits `df`, `n`, `sigma`.\
4. **Cell 4** **depends on** `df` → emits `desc`, `corr`.\
5. **Cell 5** **depends on** (`n`, `sigma`, `corr`) → renders dynamic markdown.\
6. **Cell 6** **depends on** (`desc`, `corr`) → displays tables.\
7. **Cell 7** **depends on** `df` → shows reactive scatter plot.
        """
    )

if __name__ == "__main__":
    app.run()
