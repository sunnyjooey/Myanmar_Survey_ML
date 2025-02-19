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

df = spark.sql('SELECT * FROM myanmar_ml.lcvr_ref_model_2017_fixed')
mod_df = df.toPandas()
mod_df = mod_df.loc[:, ['lat', 'lon', 'y0', 'y0_bin']].copy()

# COMMAND ----------

shp = gpd.read_file('/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adm3_shapefile/mmr_polbnda_adm3_250k_mimu_1.shp')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply models to the rest of the country

# COMMAND ----------

# get model runs and results
id = 389746355820867
max_results = 1000  # Maximum number of rows to retrieve
runs = mlflow.search_runs(experiment_ids=[id], max_results=max_results)
runs['prec-recall-diff'] = abs(runs['metrics.test_precision_score'] - runs['metrics.test_recall_score'])

# COMMAND ----------

# MAGIC %md
# MAGIC best eye-balled models: receptive-gnat, kindly-vol, judicious-wolf, debonair-hog

# COMMAND ----------

runs.loc[(runs['metrics.test_f1_score'] > 0.4) & (runs['prec-recall-diff'] < 0.1), ['tags.mlflow.runName','metrics.test_f1_score']].sort_values('metrics.test_f1_score', ascending=False)

# COMMAND ----------

# good f1 score, low difference precision and recall
list(runs.loc[(runs['metrics.test_f1_score'] > 0.4) & (runs['prec-recall-diff'] < 0.1), 'tags.mlflow.runName'])

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

target_col = "y0_bin"

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

def show_map(merge_data, one_run, mod, pov_diff_coord, pov_diff_bin):
    poor = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'prop_poor'].iloc[0], 2)
    auc = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_roc_auc'].iloc[0], 2)
    rec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_recall_score'].iloc[0], 2)
    prec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_precision_score'].iloc[0], 2)
    f1 = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_f1_score'].iloc[0], 2)
    title = f'{mod}\n poor {poor}; pov-diff-coord {pov_diff_coord}; pov-diff-bin {pov_diff_bin}\n roc-auc {auc}; recall {rec}; precision {prec}; f1 {f1}'
    ax = merge_data.plot(column=mod, legend=True, vmax=1.0, cmap='viridis_r')
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

mods = ['calm-sponge-868', 'omniscient-kit-369', 'bald-snipe-799', 'kindly-vole-791']

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
    
    # join
    viz_merge = gpd.tools.sjoin_nearest(viz_gdf, shp_crs, how='inner')
    mn = viz_merge[['TS', run_name]].groupby('TS').mean()
    mg = pd.merge(shp_crs, mn, on='TS', how='outer')

    ### proportion poor ###
    geometry2 = [Point(xy)  for xy in zip(mod_df['lon'], mod_df['lat'])]
    mod_gdf = gpd.GeoDataFrame(mod_df, crs=shp.crs, geometry=geometry2)

    # Project PrioGrid and Admin1 to Mercator
    mod_gdf = mod_gdf.to_crs(epsg=3857)
    shp_crs = shp.to_crs(epsg=3857)

    # join
    mod_merge = gpd.tools.sjoin_nearest(mod_gdf, shp_crs, how='inner')
    mod_mn = mod_merge[['TS', 'y0', 'y0_bin']].groupby('TS').mean()
    mod_mg = pd.merge(shp_crs, mod_mn, on='TS', how='outer')

    # proportion poor - by coordinates and survey
    pov = pd.merge(mod_mg[['TS', 'y0', 'y0_bin']], mg[['TS',run_name]], how='inner')
    pov['pov-diff-coord'] = pov['y0'] - pov[run_name]
    pov['pov-diff-bin'] = pov['y0_bin'] - pov[run_name]
    pov['pov-diff-coord-abs'] = abs(pov['pov-diff-coord'])
    pov['pov-diff-bin-abs'] = abs(pov['pov-diff-bin'])
    pov_diff_coord = round(pov['pov-diff-coord'].mean(), 2)
    pov_diff_bin = round(pov['pov-diff-bin'].mean(), 2)
    
    ### show visuals ###
    # proportion poor
    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()
    ax1 = mod_mg.plot(ax=axs[0], column='y0', legend=True, vmax=1.0, cmap='viridis_r')
    ax1.set_title('Poor-coordinates')
    ax1.set_axis_off()
    ax2 = mod_mg.plot(ax=axs[1], column='y0_bin', legend=True, vmax=1.0, cmap='viridis_r')
    ax2.set_title('Poor-survey')
    ax2.set_axis_off()

    # map of applied model
    show_map(mg, one_run_apply, run_name, pov_diff_coord, pov_diff_bin)
    
    # table of large proportion poor discrepancies
    display(pov.loc[pov['pov-diff-coord-abs'] > .5, :])
    display(pov.loc[pov['pov-diff-bin-abs'] > .5, :])

    # model evaluation
    show_graphs(runs, run_name)

# COMMAND ----------

show_model(runs, 'flawless-bass-261', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'receptive-gnat-595', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'kindly-vole-791', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'judicious-wolf-748', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'debonair-hog-713', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'honorable-doe-477', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'gentle-calf-530', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'nimble-dolphin-753', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'bedecked-smelt-108', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'loud-skunk-450', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'enchanting-hare-489', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'secretive-stag-809', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'rebellious-skink-322', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'rare-calf-969', df, app_df, coords, shp, mod_df)

# COMMAND ----------

show_model(runs, 'secretive-vole-787', df, app_df, coords, shp, mod_df)

# COMMAND ----------

run_name = 'calm-sponge-868'
show_model(runs, run_name, df, app_df, coords, shp, mod_df)

# COMMAND ----------

# for mod in ['calm-sponge-868-manual'] + mods:
#     poor = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'prop_poor'].iloc[0], 2)
#     auc = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_roc_auc'].iloc[0], 2)
#     rec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_recall_score'].iloc[0], 2)
#     prec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_precision_score'].iloc[0], 2)
#     f1 = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_f1_score'].iloc[0], 2)
#     title = f'{mod}\n poor {poor}; roc-auc {auc}; recall {rec}; precision {prec}; f1 {f1}'
#     ax = mg.plot(column=mod, legend=True, vmax=1.0, cmap='viridis_r')
#     ax.set_title(title)
#     ax.set_axis_off()
