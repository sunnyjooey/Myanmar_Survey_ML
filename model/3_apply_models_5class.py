# Databricks notebook source
!pip install geopandas

# COMMAND ----------

import pandas as pd
import re
import os
import uuid
import geopandas as gpd  
from shapely.geometry import Point
import matplotlib.pyplot as plt
from IPython.display import Image

import mlflow
import pyspark.sql.functions as F

# COMMAND ----------

df = spark.sql('SELECT * FROM myanmar_ml.lcvr_ref_model_2017_fixed_quintile')
mod_df = df.toPandas()
mod_df = mod_df.loc[:, ['lat', 'lon', 'wealth_quintile']].copy()

# COMMAND ----------

shp = gpd.read_file('/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adm3_shapefile/mmr_polbnda_adm3_250k_mimu_1.shp')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply models to the rest of the country

# COMMAND ----------

# get model runs and results
id = 2187645800560424
max_results = 1000  # Maximum number of rows to retrieve
runs = mlflow.search_runs(experiment_ids=[id], max_results=max_results)

# COMMAND ----------

f1 = runs[runs['metrics.test_f1_score']==max(runs['metrics.test_f1_score'])]
roc = runs[runs['metrics.test_roc_auc']==max(runs['metrics.test_roc_auc'])]
prec = runs[runs['metrics.test_precision_score']==max(runs['metrics.test_precision_score'])]
recall = runs[runs['metrics.test_recall_score']==max(runs['metrics.test_recall_score'])]

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

target_col = "wealth_quintile"

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

def show_map(merge_data, one_run, mod, label):
    poor = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'prop_poor'].iloc[0], 2)
    auc = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_roc_auc'].iloc[0], 2)
    rec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_recall_score'].iloc[0], 2)
    prec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_precision_score'].iloc[0], 2)
    f1 = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_f1_score'].iloc[0], 2)
    title = f'{label}\n poor {poor}; roc-auc {auc}; recall {rec}; precision {prec}; f1 {f1}'
    ax = merge_data.plot(column=0, legend=True, vmax=1.0, cmap='viridis_r')
    ax.set_title(title)
    ax.set_axis_off()

# COMMAND ----------

def show_graphs(runs, mod):
    # Create temp directory to download MLflow model artifact
    eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
    os.makedirs(eval_temp_dir, exist_ok=True)

    # Download the artifact
    run_id = runs.loc[runs['tags.mlflow.runName']==mod, 'run_id'].iloc[0]
    eval_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=eval_temp_dir)
    
    # display
    eval_confusion_matrix_path = os.path.join(eval_path, "test_confusion_matrix.png")
    display(Image(filename=eval_confusion_matrix_path, width=400))

    eval_confusion_matrix_path = os.path.join(eval_path, "test_precision_recall_curve_plot.png")
    display(Image(filename=eval_confusion_matrix_path, width=400))

    eval_confusion_matrix_path = os.path.join(eval_path, "test_roc_curve_plot.png")
    display(Image(filename=eval_confusion_matrix_path, width=400))

# COMMAND ----------

def show_model(runs, run_name, df, app_df, coords, shp, mod_df):
    print(run_name.upper())
    
    ### applied model ###
    pred_apply, one_run_apply = apply_model(runs, run_name, app_df)

    # create data for visualization
    app_df[run_name] = pred_apply
    viz = pd.concat([coords, app_df], axis=1)

    # geometry points
    geometry = [Point(xy)  for xy in zip(viz['lon'], viz['lat'])]
    viz_gdf = gpd.GeoDataFrame(viz, crs=shp.crs, geometry=geometry)

    # Project PrioGrid and Admin1 to Mercator
    viz_gdf = viz_gdf.to_crs(epsg=3857)
    shp_crs = shp.to_crs(epsg=3857)
    shp_crs1 = shp.to_crs(epsg=3857)
    shp_crs2 = shp.to_crs(epsg=3857)
    shp_crs1n2 = shp.to_crs(epsg=3857)
    
    # join
    viz_merge = gpd.tools.sjoin_nearest(viz_gdf, shp_crs, how='inner')
    
    # calculate proportion class 1 or 2
    class_proportions = (viz_merge[[run_name,'TS']].groupby(['TS',run_name]).size() / viz_merge[[run_name,'TS']].groupby(['TS',run_name]).size().groupby('TS').sum()).reset_index()
    class1 = class_proportions[class_proportions[run_name]=='1']
    class2 = class_proportions[class_proportions[run_name]=='2']
    class1n2 = class_proportions[(class_proportions[run_name]=='2') | (class_proportions[run_name]=='1')]
    class1n2 = class1n2.groupby('TS').sum().reset_index()

    # merge with shapefile
    mg1 = pd.merge(shp_crs1, class1, on='TS', how='outer')
    mg2 = pd.merge(shp_crs2, class2, on='TS', how='outer')
    mg1n2 = pd.merge(shp_crs1n2, class1n2, on='TS', how='outer')

    # map of applied model
    show_map(mg1, one_run_apply, run_name, 'Quint-1')
    show_map(mg2, one_run_apply, run_name, 'Quint-2')
    show_map(mg1n2, one_run_apply, run_name, 'Quint-1&2')

    ### proportion poor ###
    geometry2 = [Point(xy)  for xy in zip(mod_df['lon'], mod_df['lat'])]
    mod_gdf = gpd.GeoDataFrame(mod_df, crs=shp.crs, geometry=geometry2)

    # # Project PrioGrid and Admin1 to Mercator
    mod_gdf = mod_gdf.to_crs(epsg=3857)
    shp_crs = shp.to_crs(epsg=3857)
    shp_crs1 = shp.to_crs(epsg=3857)
    shp_crs2 = shp.to_crs(epsg=3857)
    shp_crs1n2 = shp.to_crs(epsg=3857)

    # join
    mod_merge = gpd.tools.sjoin_nearest(mod_gdf, shp_crs, how='inner')
    
    # calculate proportion class 1 or 2
    class_proportions = (mod_merge[['wealth_quintile','TS']].groupby(['TS','wealth_quintile']).size() / mod_merge[['wealth_quintile','TS']].groupby(['TS','wealth_quintile']).size().groupby('TS').sum()).reset_index()
    class1 = class_proportions[class_proportions['wealth_quintile']=='1']
    class2 = class_proportions[class_proportions['wealth_quintile']=='2']
    class1n2 = class_proportions[(class_proportions['wealth_quintile']=='2') | (class_proportions['wealth_quintile']=='1')]
    class1n2 = class1n2.groupby('TS').sum().reset_index()

    # merge with shapefile
    md1 = pd.merge(shp_crs1, class1, on='TS', how='outer')
    md2 = pd.merge(shp_crs2, class2, on='TS', how='outer')
    md1n2 = pd.merge(shp_crs1n2, class1n2, on='TS', how='outer')

    # map of applied model
    show_map(md1, one_run_apply, run_name, 'Quint-1')
    show_map(md2, one_run_apply, run_name, 'Quint-2')
    show_map(md1n2, one_run_apply, run_name, 'Quint-1&2')

    # model evaluation
    show_graphs(runs, run_name)

# COMMAND ----------

show_model(runs, 'exultant-fox-81', df, app_df, coords, shp, mod_df)

# COMMAND ----------


