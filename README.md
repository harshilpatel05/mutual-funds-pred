# Mutual Funds Recommendation (Low-Memory Pipeline)

This project builds a **memory-efficient mutual fund analytics and recommendation system** from NAV history, then produces:
- top similar funds (cosine similarity in feature space)
- diversification candidates (low-correlation funds)
- risk-aware ranking and clustering
- ablation comparison between simple vs full feature sets

The latest implementation is in **`model2.ipynb`**.

---

## What changed in the new approach

The updated notebook uses a low-memory, production-style pipeline:

1. **Load only required columns** from `final_data.csv`.
2. **Clean + optimize dtypes** (`int32`, `float32`, `category`) via `prepare_nav_df_lowmem`.
3. **Compute returns efficiently** using grouped transformations (`add_returns_lowmem`).
4. **Build robust fund-level features** (`compute_fund_features_lowmem`), including:
   - CAGR, annualized volatility, Sharpe, Sortino
   - max drawdown, ulcer index, VaR/CVaR
   - skew, kurtosis, trend slope of log NAV
5. **Incorporate precomputed technical columns** if present (`aggregate_precomputed_features_lowmem`).
6. **Normalize and vectorize** features (`fit_feature_matrix`) for fast cosine similarity.
7. **Generate recommendations + diagnostics**:
   - Similar fund retrieval (`recommend_similar_topk`)
   - Diversification proxy (`recommend_diversifiers_proxy`)
   - PCA + KMeans visualization (`plot_feature_space_clusters`)
   - Full case report (`case_study_report`)
   - Ablation analysis (`ablation_similarity_comparison_lowmem`)

---

## Visual results from the updated notebook

### 1) NAV Case Study (Target vs Similar Funds)

Shows the target fund NAV trajectory against its top similar funds.

![NAV Case Study](PastTrends.png)

---

### 2) Correlation Matrix (Portfolio + Candidates)

Used to evaluate diversification quality (prefer lower correlation with current portfolio exposure).

![Correlation Matrix](CorrelationMatrix.png)

---

### 3) Feature Space Clusters (PCA)

Projects high-dimensional features to 2D and groups funds into clusters for structural understanding.

![Feature Space Clusters](FeatureSpace.png)

---

### 4) Ablation Study (Simple vs Full Features)

Compares recommendation behavior between:
- **Simple model:** core risk-return features
- **Full model:** enriched feature set including tail risk/trend/precomputed indicators

![Ablation Study](abltationStudies.png)

---

## Quick run (from notebook)

Open `model2.ipynb` and run cells in order. The main execution flow is:

```python
art = build_recommenders_lowmem(maindf, min_obs=100, build_wide_returns=True)

report = case_study_report(
    art,
    target_scheme_code=100027,
    portfolio_codes=[100027, 123456],
    top_k_similar=5,
    nav_window=None
)

if report["diversifiers"] is not None:
    show_codes = [100027, 123456] + report["diversifiers"]["schemeCode"].tolist()
    corr = plot_correlation_heatmap(art["wide_returns"], show_codes)

cluster_df, pca, km = plot_feature_space_clusters(art, n_clusters=8)
abl = ablation_similarity_comparison_lowmem(art, target_code=100027, k=10)
tables = paper_tables(art, top_n=20)
```

---

## Project files

- `model2.ipynb` → latest full low-memory pipeline + plots
- `final_data.csv` → prepared NAV dataset used by notebook
- `rf.joblib`, `xgbr.joblib`, `features_meta.joblib` → saved artifacts
- `PastTrends.png`, `CorrelationMatrix.png`, `FeatureSpace.png`, `abltationStudies.png` → result figures

---

## Notes

- The recommendation engine is **feature-similarity based** (not direct return forecasting).
- Similarity quality depends on data freshness and feature completeness.
- Use realistic `portfolio_codes` available in your dataset for diversification analysis.
