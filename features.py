"""Preparación de datos y feature engineering (compartido por train.py y app.py)."""
import numpy as np
import pandas as pd

# Columnas donde NA significa "no tiene" (según data_description.txt)
NA_ES_NONE = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
              "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
              "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
              "MasVnrType"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + variables derivadas. No usa información del target."""
    d = df.copy()
    d = d.drop(columns=[c for c in ["Id", "SalePrice"] if c in d.columns])
    for c in NA_ES_NONE:
        if c in d.columns:
            d[c] = d[c].fillna("None")
    d["MSSubClass"] = d["MSSubClass"].astype(str)          # es categórica
    d["MoSold"] = d["MoSold"].astype(str)
    # Feature engineering
    d["TotalSF"] = d["TotalBsmtSF"].fillna(0) + d["1stFlrSF"] + d["2ndFlrSF"]
    d["TotalBath"] = (d["FullBath"] + 0.5 * d["HalfBath"]
                      + d["BsmtFullBath"].fillna(0) + 0.5 * d["BsmtHalfBath"].fillna(0))
    d["HouseAge"] = (d["YrSold"] - d["YearBuilt"]).clip(lower=0)
    d["RemodAge"] = (d["YrSold"] - d["YearRemodAdd"]).clip(lower=0)
    d["TotalPorchSF"] = (d["OpenPorchSF"] + d["EnclosedPorch"] + d["3SsnPorch"]
                         + d["ScreenPorch"] + d["WoodDeckSF"])
    d["HasPool"] = (d["PoolArea"] > 0).astype(int)
    d["HasGarage"] = (d["GarageArea"].fillna(0) > 0).astype(int)
    d["HasFireplace"] = (d["Fireplaces"] > 0).astype(int)
    d["QualxSF"] = d["OverallQual"] * d["GrLivArea"]
    d = d.drop(columns=["GarageYrBlt", "YrSold"])
    return d
