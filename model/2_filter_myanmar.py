# Databricks notebook source
!pip install geopandas

# COMMAND ----------

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from pyspark.sql.functions import monotonically_increasing_id

# COMMAND ----------

df = spark.sql("SELECT * FROM myanmar_ml.lcvr_ref_apply_2017")

# COMMAND ----------

columns_to_drop = ['y0', 'y0_nw', 'ya_25', 'ya_50', 'ya_75', 'ya_50_nw', 'y0_bin', 'ya_25_bin', 'ya_50_bin', 'ya_75_bin']
df = df.drop(*columns_to_drop)

# COMMAND ----------

# This will return a new DF with all the columns + id
df = df.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

mm = gpd.read_file('/dbfs/FileStore/Myanmar_Survey_ML/data/geo/country_shapefile/mmr_polbnda_adm0_250k_mimu_1.shp')

# COMMAND ----------

mx = df.count() + 1
bn = 1000000
rng = int(np.ceil(mx / bn) + 1)
tot_rows = 0

start_i = 0
end_i = bn + 1
for i in range(1, rng):
    print(start_i, end_i)
    df_sub = df.filter((df['id']>=start_i) & (df['id']<end_i))
    df_sub = df_sub.toPandas()
    
    geometry = [Point(xy)  for xy in zip(df_sub['lon'], df_sub['lat'])]
    adm_gdf = gpd.GeoDataFrame(df_sub, crs=mm.crs, geometry=geometry)
    points_within = gpd.sjoin(adm_gdf, mm, how='inner', predicate='within')
    points_within = points_within.drop(['id', 'geometry', 'index_right', 'OBJECTID', 'Name'], axis=1)
    nrows = points_within.shape[0]
    tot_rows += nrows
    print(tot_rows)

    if nrows > 0: 
        points_within = spark.createDataFrame(points_within)
        points_within.write.mode('append').format('delta').saveAsTable('myanmar_ml.lcvr_ref_apply_2017_fixed')

    start_i = end_i - 1
    end_i += bn
    if end_i > mx:
        end_i = mx

# COMMAND ----------


