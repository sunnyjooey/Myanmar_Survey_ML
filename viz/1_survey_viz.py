# Databricks notebook source
# MAGIC %md
# MAGIC ### Check survey response - proportion poor

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

poverty_line = 1302.951
weight = 'hhweight'
expenditure = 'r_totex_pad_v3'

# COMMAND ----------

# MAGIC %md
# MAGIC NOTE: try proportion poor with weights

# COMMAND ----------

# read in survey
surv = pd.read_stata('/dbfs/FileStore/Myanmar_Survey_ML/data/survey/Assets_household_level.dta')
# create poverty y and y-binary
surv = pd.concat([surv, pd.Series((poverty_line - surv[expenditure]).clip(0) / poverty_line, name='y')], axis=1)
surv['ybin'] = surv['y'].apply(lambda x: 1 if x > 0 else 0)

# COMMAND ----------

sum(surv['poor'] != surv['ybin'])

# COMMAND ----------

surv.groupby('wealth_quintile').size()

# COMMAND ----------

# proportion poor (1=poor)
surv.groupby('ybin').size() / surv.groupby('ybin').size().sum()

# COMMAND ----------

# read in survey merged with predictor data (dataset used for modeling)
df = spark.sql("SELECT * FROM myanmar_ml.lcvr_ref_model_2017_fixed")

# COMMAND ----------


# proportion poor (1=poor)
tot = df.count()
df.groupBy('y0_bin').count().withColumnRenamed('count', 'cnt_per_group').withColumn('perc_of_count_total', F.col('cnt_per_group') / tot ).show()

# COMMAND ----------

!pip install geopandas

# COMMAND ----------

import geopandas as gpd  
from shapely.geometry import Point

# COMMAND ----------

shp = gpd.read_file('/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adm3_shapefile/mmr_polbnda_adm3_250k_mimu_1.shp')

# COMMAND ----------

geometry = [Point(xy)  for xy in zip(surv['s0q23'], surv['s0q22'])]
surv_gdf = gpd.GeoDataFrame(surv, crs=shp.crs, geometry=geometry)

MercatorProjCode = 3857
WSG84CRSCode = 4326
       
# Project PrioGrid and Admin1 to Mercator
surv_gdf = surv_gdf.to_crs(epsg=MercatorProjCode)
shp = shp.to_crs(epsg=MercatorProjCode)

# COMMAND ----------

fig, ax = plt.subplots()
shp.plot(ax=ax, alpha=0.4, color='grey')
surv_gdf.plot(ax=ax, 
            markersize=.5, 
            color='blue', 
            marker='o')

# COMMAND ----------

# join
surv_merge = gpd.tools.sjoin_nearest(surv_gdf, shp, how='inner')
surv_mn = surv_merge[['ybin', 'TS']].groupby('TS').mean()
surv_mg = pd.merge(shp, surv_mn, on='TS', how='outer')
surv_mg.plot(column='ybin', legend=True, vmax=1.0, cmap='viridis_r')

# COMMAND ----------


