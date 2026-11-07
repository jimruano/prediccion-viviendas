"""Entrenamiento y evaluación de los 4 modelos (CRISP-DM: fases 3, 4 y 5)."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import prepare

SEED = 42
raw = pd.read_csv("data/train.csv")

# Outliers documentados por el autor del dataset (casas enormes vendidas baratas)
raw = raw[~((raw["GrLivArea"] > 4000) & (raw["SalePrice"] < 300000))].reset_index(drop=True)
X = prepare(raw)
y = raw["SalePrice"]
y_log = np.log1p(y)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


models = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_features=0.33,
                                           min_samples_leaf=1, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=1000, learning_rate=0.03,
                                                   max_depth=3, subsample=0.8,
                                                   max_features=0.3, random_state=SEED),
    "Red Neuronal (MLP)": MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1.0,
                                       learning_rate_init=1e-3, max_iter=1500,
                                       early_stopping=True, random_state=SEED),
}

X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=SEED)
results, fitted, preds = {}, {}, {}
for name, est in models.items():
    pipe = Pipeline([("pre", make_pre()), ("model", clone(est))]).fit(X_tr, y_tr)
    p = np.expm1(pipe.predict(X_te))
    real = np.expm1(y_te)
    results[name] = {
        "RMSE": float(np.sqrt(mean_squared_error(real, p))),
        "MAE": float(mean_absolute_error(real, p)),
        "R2": float(r2_score(real, p)),
        "RMSLE": float(np.sqrt(mean_squared_error(y_te, np.log1p(p)))),
    }
    fitted[name], preds[name] = pipe, p
    print(name, {k: round(v, 4) for k, v in results[name].items()})

best = min(results, key=lambda k: results[k]["RMSE"])
print("Mejor modelo:", best)

# Reentrenar cada modelo con TODOS los datos para despliegue
final = {n: Pipeline([("pre", make_pre()), ("model", clone(e))]).fit(X, y_log)
         for n, e in models.items()}
joblib.dump(final, "models/models.joblib", compress=("xz", 6))

# Importancia de variables (permutación sobre variables originales, mejor modelo)
imp = permutation_importance(fitted[best], X_te, y_te, n_repeats=8,
                             random_state=SEED, n_jobs=-1,
                             scoring="neg_root_mean_squared_error")
imp_df = (pd.DataFrame({"variable": X.columns, "importancia": imp.importances_mean})
          .sort_values("importancia", ascending=False).head(15))
imp_df.to_csv("models/importancias.csv", index=False)

# Metadatos para la app (valores por defecto y opciones)
meta = {
    "best": best, "metrics": results,
    "num_defaults": {c: float(raw_c) for c, raw_c in
                     X[num_cols].median().items()},
    "cat_defaults": {c: X[c].mode()[0] for c in cat_cols},
    "cat_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
    "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in num_cols},
    "neigh_median": raw.groupby("Neighborhood")["SalePrice"].median().to_dict(),
    "columns": X.columns.tolist(),
}
json.dump(meta, open("models/meta.json", "w"), indent=1, default=str)

# Figuras para el documento
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(4.2, 3))
imp_df.iloc[::-1].plot.barh(x="variable", y="importancia", ax=ax, legend=False, color="#1f4e79")
ax.set_xlabel("Aumento del RMSE (log) al permutar"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig("figures/importancia.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(3.4, 3))
ax.scatter(np.expm1(y_te) / 1e3, preds[best] / 1e3, s=8, alpha=.6, color="#1f4e79")
m = np.expm1(y_te).max() / 1e3
ax.plot([0, m], [0, m], "r--", lw=1)
ax.set_xlabel("Precio real (miles USD)"); ax.set_ylabel("Precio predicho (miles USD)")
ax.set_title(best)
plt.tight_layout(); plt.savefig("figures/real_vs_pred.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.2, 2.4))
ax.hist(y / 1e3, bins=40, color="#1f4e79", alpha=.8, label="SalePrice")
ax.set_xlabel("Precio de venta (miles USD)"); ax.set_ylabel("Frecuencia")
plt.tight_layout(); plt.savefig("figures/distribucion.png", dpi=200); plt.close()
