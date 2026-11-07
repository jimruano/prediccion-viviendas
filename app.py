"""Aplicación web de predicción de precios de viviendas (CRISP-DM: fase 6, Despliegue).
Ejecutar:  streamlit run app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import prepare

st.set_page_config(page_title="Predicción de precio de viviendas", page_icon="🏠", layout="wide")


@st.cache_resource
def load():
    return joblib.load("models/models.joblib"), json.load(open("models/meta.json"))


models, meta = load()
raw = pd.read_csv("data/train.csv")
imp = pd.read_csv("models/importancias.csv")

# Fila base con valores típicos (mediana / moda) sobre las columnas originales
base = raw.drop(columns=["Id", "SalePrice"]).median(numeric_only=True).to_dict()
for c in raw.columns.drop(["Id", "SalePrice"]):
    if c not in base:
        base[c] = raw[c].mode()[0] if raw[c].notna().any() else "None"


def predict(overrides: dict, model_name: str) -> float:
    row = {**base, **overrides}
    X = prepare(pd.DataFrame([row]))
    return float(np.expm1(models[model_name].predict(X)[0]))


st.title("🏠 Predicción del precio de viviendas (Ames, Iowa)")
st.caption("Proyecto final de Minería de Datos · metodología CRISP-DM")

tab1, tab2, tab3, tab4 = st.tabs(["Predicción", "Variables importantes",
                                  "Simulador de precios", "Modelos y métricas"])

# --------------------------------------------------------------- Predicción
with tab1:
    model_name = st.selectbox("Modelo", list(models), index=list(models).index(meta["best"]))
    c1, c2, c3 = st.columns(3)
    with c1:
        neigh = st.selectbox("Barrio", sorted(raw["Neighborhood"].unique()), index=5)
        qual = st.slider("Calidad general (1-10)", 1, 10, 6)
        cond = st.slider("Condición general (1-10)", 1, 10, 5)
        year = st.slider("Año de construcción", 1872, 2010, 1990)
    with c2:
        area = st.number_input("Área habitable sobre el suelo (pies²)", 300, 6000, 1500, 50)
        bsmt = st.number_input("Sótano total (pies²)", 0, 6000, 1000, 50)
        lot = st.number_input("Tamaño del lote (pies²)", 1000, 60000, 9000, 500)
        stories2 = st.number_input("Segundo piso (pies²)", 0, 2500, 0, 50)
    with c3:
        full = st.slider("Baños completos", 0, 4, 2)
        bed = st.slider("Habitaciones", 0, 8, 3)
        gcars = st.slider("Capacidad del garaje (autos)", 0, 4, 2)
        kitchen = st.selectbox("Calidad de cocina", ["Ex", "Gd", "TA", "Fa"], index=2)
    ov = {"Neighborhood": neigh, "OverallQual": qual, "OverallCond": cond, "YearBuilt": year,
          "YearRemodAdd": year,
          "GrLivArea": area, "1stFlrSF": max(area - stories2, 300), "2ndFlrSF": stories2,
          "TotalBsmtSF": bsmt, "LotArea": lot, "FullBath": full, "BedroomAbvGr": bed,
          "GarageCars": gcars, "GarageArea": gcars * 280, "KitchenQual": kitchen}
    price = predict(ov, model_name)
    st.metric("Precio estimado", f"US$ {price:,.0f}")
    mae = meta["metrics"][model_name]["MAE"]
    st.caption(f"Error absoluto medio del modelo en prueba: ±US$ {mae:,.0f}")

# ------------------------------------------------------------- Importancias
with tab2:
    st.subheader("Variables que más influyen en el precio (importancia por permutación)")
    st.bar_chart(imp.set_index("variable")["importancia"].sort_values())
    st.caption("Mide cuánto empeora el error del modelo al desordenar cada variable. "
               "QualxSF = calidad × área habitable; TotalSF = sótano + 1.º y 2.º piso.")

# --------------------------------------------------------------- Simulador
with tab3:
    st.subheader("¿Cuánto valor aporta cada mejora?")
    st.write("Parte de la vivienda definida en la pestaña *Predicción* y simula cambios.")
    s1, s2 = st.columns(2)
    with s1:
        d_qual = st.slider("Subir calidad general en…", 0, 3, 1)
        d_bath = st.slider("Agregar baños completos", 0, 2, 0)
        d_sf = st.slider("Ampliar área habitable (pies²)", 0, 1000, 0, 50)
    with s2:
        d_gar = st.slider("Agregar capacidad de garaje (autos)", 0, 2, 0)
        d_kit = st.selectbox("Remodelar cocina a", ["(sin cambio)", "TA", "Gd", "Ex"])
        d_pool = st.checkbox("Agregar piscina (500 pies²)")
    sim = dict(ov)
    sim["OverallQual"] = min(10, qual + d_qual)
    sim["FullBath"] = full + d_bath
    sim["GrLivArea"] = area + d_sf
    sim["1stFlrSF"] = ov["1stFlrSF"] + d_sf
    sim["GarageCars"] = min(4, gcars + d_gar)
    sim["GarageArea"] = sim["GarageCars"] * 280
    if d_kit != "(sin cambio)":
        sim["KitchenQual"] = d_kit
    if d_pool:
        sim.update({"PoolArea": 500, "PoolQC": "Gd"})
    new_price = predict(sim, model_name)
    k1, k2, k3 = st.columns(3)
    k1.metric("Precio actual", f"US$ {price:,.0f}")
    k2.metric("Precio simulado", f"US$ {new_price:,.0f}", f"{new_price - price:+,.0f}")
    k3.metric("Variación", f"{(new_price / price - 1) * 100:+.1f} %")

    st.markdown("**Precio vs. calidad general (resto igual)**")
    curve = pd.DataFrame({"Calidad": range(1, 11),
                          "Precio": [predict({**ov, "OverallQual": q}, model_name) for q in range(1, 11)]})
    st.line_chart(curve.set_index("Calidad"))

# ------------------------------------------------------------------ Métricas
with tab4:
    st.subheader("Comparación de modelos (20 % de datos de prueba)")
    mt = pd.DataFrame(meta["metrics"]).T[["RMSE", "MAE", "R2"]]
    st.dataframe(mt.style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.4f}"}))
    st.caption(f"Mejor modelo por RMSE: **{meta['best']}**")
