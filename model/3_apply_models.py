# Databricks notebook source
import pandas as pd
import mlflow
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check survey response

# COMMAND ----------

poverty_line = 1302.951
weight = 'hhweight'
expenditure = 'r_totex_pad_v3'

# COMMAND ----------

# read in survey
surv = pd.read_stata('/dbfs/FileStore/Myanmar_Survey_ML/data/survey/Assets_household_level.dta')
# create poverty y and y-binary
surv = pd.concat([surv, pd.Series((poverty_line - surv[expenditure]).clip(0) / poverty_line, name='y')], axis=1)
surv['ybin'] = surv['y'].apply(lambda x: 1 if x > 0 else 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply models to the rest of the country

# COMMAND ----------

# get model runs and results
id = 389746355820867
max_results = 1000  # Maximum number of rows to retrieve
runs = mlflow.search_runs(experiment_ids=[id], max_results=max_results)

# COMMAND ----------

# the rest of the data on which the model will be applied
app_df = spark.sql("SELECT * FROM myanmar_ml.lcvr_ref_apply_2017_fixed")
app_df = app_df.toPandas()
coords = app_df.loc[:, ['time', 'lat', 'lon']].copy()
app_df = app_df.drop(['time', 'lat', 'lon'], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #####1. Model with Highest Validation f1-score: calm-sponge-868  

# COMMAND ----------

# fitting the model super manually - refitting on entire data, then applying
# application of the model directly led to 99% coordinates poor!

target_col = "y0_bin"

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

supported_cols = ["190", "viirs", "110", "120", "30", "201", "50", "150", "151", "153", "40", "200", "72", "Qh_tavg", "160", "130", "152", "180", "70", "71", "fatal_count", "SoilMoi100_200cm_tavg", "landscan", "170", "10", "62", "event_count", "Qair_f_tavg", "SoilMoi10_40cm_tavg", "82", "SoilMoi40_100cm_tavg", "121", "Qs_tavg", "11", "Evap_tavg", "60", "Tair_f_tavg", "61", "Qg_tavg", "90", "12", "SoilMoi00_10cm_tavg", "20", "81", "100", "Rainf_f_tavg"]
col_selector = ColumnSelector(supported_cols)


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

bool_imputers = []
bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
    ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
])
bool_transformers = [("boolean", bool_pipeline, ["190", "110", "120", "30", "201", "50", "151", "153", "40", "200", "72", "160", "20", "130", "152", "180", "70", "71", "170", "10", "62", "82", "121", "11", "60", "61", "90", "12", "150", "81", "100"])]


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["10", "100", "11", "110", "12", "120", "121", "130", "150", "151", "152", "153", "160", "170", "180", "190", "20", "200", "201", "30", "40", "50", "60", "61", "62", "70", "71", "72", "81", "82", "90", "Evap_tavg", "Qair_f_tavg", "Qg_tavg", "Qh_tavg", "Qs_tavg", "Rainf_f_tavg", "SoilMoi00_10cm_tavg", "SoilMoi100_200cm_tavg", "SoilMoi10_40cm_tavg", "SoilMoi40_100cm_tavg", "Tair_f_tavg", "event_count", "fatal_count", "landscan", "viirs"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["190", "viirs", "110", "120", "30", "201", "50", "151", "153", "40", "200", "72", "Qh_tavg", "160", "130", "20", "152", "180", "70", "71", "fatal_count", "SoilMoi100_200cm_tavg", "landscan", "170", "event_count", "10", "62", "Qair_f_tavg", "SoilMoi10_40cm_tavg", "82", "SoilMoi40_100cm_tavg", "121", "Qs_tavg", "11", "Evap_tavg", "60", "Tair_f_tavg", "Qg_tavg", "61", "90", "12", "SoilMoi00_10cm_tavg", "150", "81", "100", "Rainf_f_tavg"])]


from sklearn.compose import ColumnTransformer

transformers = bool_transformers + numerical_transformers
preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)


import lightgbm
from lightgbm import LGBMClassifier

space = {
  "colsample_bytree": 0.42792779427916483,
  "lambda_l1": 135.74281049758403,
  "lambda_l2": 11.394551336323014,
  "learning_rate": 2.930187734854735,
  "max_bin": 475,
  "max_depth": 8,
  "min_child_samples": 244,
  "n_estimators": 211,
  "num_leaves": 32,
  "path_smooth": 26.75429763700822,
  "subsample": 0.6717962717560846,
  "random_state": 824288948,
}


df = df.toPandas()
y_train = df[target_col]
drop = ['time', 'lat', 'lon', 'y0', 'y0_nw', 'ya_25', 'ya_50', 'ya_75', 'ya_50_nw', 'y0_bin', 'ya_25_bin', 'ya_50_bin', 'ya_75_bin']
X_train = df.drop(drop, axis=1)


pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
lgbmc_classifier = LGBMClassifier(**space)
model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("classifier", lgbmc_classifier),
])
model.fit(X_train, y_train)

run_name = 'calm-sponge-868'
model_name = f'{run_name}-manual'
# predict
pred = model.predict(app_df)
app_df[model_name] = pred

# proportion poor
one_run = runs.loc[runs['tags.mlflow.runName']==run_name, ['run_id', 'tags.mlflow.runName', 'metrics.test_f1_score', 'metrics.test_precision_score', 'metrics.test_recall_score', 'metrics.test_roc_auc']]
one_run.loc[one_run['tags.mlflow.runName']==run_name, 'tags.mlflow.runName'] = model_name
proportions = app_df.groupby(model_name).size() / app_df.groupby(model_name).size().sum()
one_run['prop_poor'] = [proportions[1]]

# COMMAND ----------

def apply_model(runs_df, run_name, apply_df):
    # Get scores for one model run
    one_run = runs.loc[runs['tags.mlflow.runName']==run_name, ['run_id', 'tags.mlflow.runName', 'metrics.test_f1_score', 'metrics.test_precision_score', 'metrics.test_recall_score', 'metrics.test_roc_auc']]
    run_id = one_run['run_id'].iloc[0]

    # Load model as a PyFuncModel.
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    pred = loaded_model.predict(apply_df)
    apply_df[run_name] = pred

    proportions = app_df.groupby(run_name).size() / app_df.groupby(run_name).size().sum()
    one_run['prop_poor'] = [proportions[1]]

    return pred, one_run


# COMMAND ----------

# MAGIC %md
# MAGIC Run model predictions

# COMMAND ----------

mods = ['calm-sponge-868', 'omniscient-kit-369', 'bald-snipe-799', 'kindly-vole-791']
for mod in mods:
    pred, next_one_run = apply_model(runs, mod, app_df)
    app_df[mod] = pred
    one_run = pd.concat([one_run, next_one_run])

# COMMAND ----------

viz = pd.concat([coords, app_df[[model_name] + mods]], axis=1)

# COMMAND ----------

!pip install geopandas

# COMMAND ----------

import geopandas as gpd  
from shapely.geometry import Point

# COMMAND ----------

shp = gpd.read_file('/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adm3_shapefile/mmr_polbnda_adm3_250k_mimu_1.shp')

# COMMAND ----------

geometry = [Point(xy)  for xy in zip(viz['lon'], viz['lat'])]
viz_gdf = gpd.GeoDataFrame(viz, crs=shp.crs, geometry=geometry)

# COMMAND ----------

MercatorProjCode = 3857
WSG84CRSCode = 4326
       
# Project PrioGrid and Admin1 to Mercator
viz_gdf = viz_gdf.to_crs(epsg=MercatorProjCode)
shp = shp.to_crs(epsg=MercatorProjCode)
 
# join
viz_merge = gpd.tools.sjoin_nearest(viz_gdf, shp, how='inner')

# COMMAND ----------

mn = viz_merge[['TS', model_name] + mods].groupby('TS').mean()
mg = pd.merge(shp, mn, on='TS', how='outer')

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

for mod in ['calm-sponge-868-manual'] + mods:
    poor = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'prop_poor'].iloc[0], 2)
    auc = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_roc_auc'].iloc[0], 2)
    rec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_recall_score'].iloc[0], 2)
    prec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_precision_score'].iloc[0], 2)
    f1 = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_f1_score'].iloc[0], 2)
    title = f'{mod}\n poor {poor}; roc-auc {auc}; recall {rec}; precision {prec}; f1 {f1}'
    ax = mg.plot(column=mod, legend=True, vmax=1.0, cmap='viridis_r')
    ax.set_title(title)
    ax.set_axis_off()

# COMMAND ----------

# import matplotlib.pyplot as plt
# pnt_Test = viz_merge[viz_merge.ET_ID==0]
# base = shp.boundary.plot(linewidth=1, edgecolor="black")
# viz_merge.plot(ax=base, linewidth=1, color="blue", markersize=1)
# pnt_Test.plot(ax=base, linewidth=1, color="red", markersize=1)
# plt.show()

# COMMAND ----------


