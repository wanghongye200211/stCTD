#!/usr/bin/env python3
"""Create a small synthetic input for checking installation and the training CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo_input.h5ad"))
    args = parser.parse_args()
    rng = np.random.default_rng(19491001)
    time = np.repeat(np.arange(3, dtype=np.float32), 24)
    group = np.tile(np.repeat(np.arange(2), 12), 3)
    spatial = rng.normal(0, 0.15, (time.size, 2)).astype(np.float32)
    spatial[:, 0] += group + 0.15 * time
    spatial[:, 1] += 0.1 * time
    latent = rng.normal(0, 0.1, (time.size, 3)).astype(np.float32)
    latent[:, 0] += group
    latent[:, 1] += 0.2 * time
    expression = np.maximum(latent @ rng.normal(size=(3, 8)) + 2, 0).astype(np.float32)
    adata = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {"time_input": time, "time": time, "Annotation": pd.Categorical([f"type_{g}" for g in group])},
            index=[f"demo_{i}" for i in range(time.size)],
        ),
        var=pd.DataFrame(index=[f"toy_gene_{i}" for i in range(expression.shape[1])]),
    )
    adata.obsm["X_spatial_input"] = spatial
    adata.obsm["X_gene_input"] = latent
    adata.layers["lognorm"] = expression.copy()
    adata.uns["data_source"] = "Synthetic software demonstration; not a biological benchmark or GAT embedding."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Wrote {args.output}: {adata.n_obs} observations, 3 time points, 3 latent dimensions.")


if __name__ == "__main__":
    main()
