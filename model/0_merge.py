# Databricks notebook source
!pip install numpy==1.23.0
!pip install xarray

# COMMAND ----------

# restart the python kernel to import xarray correctly!
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import datetime as dt
import pickle

# COMMAND ----------

import xarray as xr

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType

# COMMAND ----------

# paths
dirpath = '/dbfs/FileStore/Myanmar_Survey_ML/data'

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Landscan

# COMMAND ----------

with open(f'{dirpath}/geo/all_x_lcvr_ref_xarray.pickle', 'rb') as handle:
    x_geo = pickle.load(handle)
    
# save each source separately (memory issue)
source = 'landscan'
x_geo = x_geo[[source]]
x_geo = x_geo.to_dataframe().dropna().reset_index().drop(['time','source'], axis=1)

# COMMAND ----------

# declare schema, change to pyspark df, save
schema = StructType([StructField("lat", FloatType(), True), StructField("lon", FloatType(), True), StructField(source, FloatType(), True)])
x_geo = spark.createDataFrame(x_geo, schema)
x_geo.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_{source}_2017')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Landcover

# COMMAND ----------

with open(f'{dirpath}/geo/all_x_lcvr_ref_xarray.pickle', 'rb') as handle:
    x_geo = pickle.load(handle)
    
# save each source separately (memory issue)
source = 'landcover'
x_geo = x_geo[[source]]
x_geo = x_geo.to_dataframe().dropna().reset_index().drop(['time','source'], axis=1)

# dummify landcover
dummies = pd.get_dummies(x_geo[source])
dummies.columns = [str(int(col)) for col in dummies.columns]

# concat together
x_geo = x_geo.drop(source, axis=1)
x_geo = pd.concat([x_geo, dummies], axis=1)

# COMMAND ----------

# declare schema, change to pyspark df, save
schema = [StructField("lat", FloatType(), True), StructField("lon", FloatType(), True)]
schema2 = [StructField(col, IntegerType(), True) for col in x_geo.columns if col not in ['lat', 'lon']]
schema.extend(schema2)
schema = StructType(schema)

x_geo = spark.createDataFrame(x_geo, schema)
x_geo.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_{source}_2017')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Viirs

# COMMAND ----------

with open(f'{dirpath}/geo/all_x_lcvr_ref_xarray.pickle', 'rb') as handle:
    x_geo = pickle.load(handle)
    
# save each source separately (memory issue)
source = 'viirs'
x_geo = x_geo[[source]]
x_geo = x_geo.to_dataframe().dropna().reset_index().drop(['source'], axis=1)

# COMMAND ----------

# declare schema, change to pyspark df, save
schema = StructType([StructField("time", DateType(), True), StructField("lat", FloatType(), True), StructField("lon", FloatType(), True), StructField(source, FloatType(), True)])
x_geo = spark.createDataFrame(x_geo, schema)
x_geo.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_{source}_2017')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### FLDAS

# COMMAND ----------

with open(f'{dirpath}/geo/all_x_lcvr_ref_xarray.pickle', 'rb') as handle:
    x_geo = pickle.load(handle)
    
# save each source separately (memory issue)
source = 'fldas'
x_vars = ['Evap_tavg', 'Qair_f_tavg','Qg_tavg', 'Qh_tavg', 'Qs_tavg', 'Rainf_f_tavg', 'SoilMoi00_10cm_tavg', 'SoilMoi100_200cm_tavg', 'SoilMoi10_40cm_tavg', 'SoilMoi40_100cm_tavg', 'Tair_f_tavg']
x_geo = x_geo[x_vars]
x_geo = x_geo.to_dataframe().dropna().reset_index().drop(['source'], axis=1)

# COMMAND ----------

# declare schema, change to pyspark df, save
schema = [StructField("time", DateType(), True), StructField("lat", FloatType(), True), StructField("lon", FloatType(), True)]
schema2 = [StructField(col, FloatType(), True) for col in x_geo.columns if col not in ['time', 'lat', 'lon']]
schema.extend(schema2)
schema = StructType(schema)

# COMMAND ----------

x_geo = spark.createDataFrame(x_geo, schema)
x_geo.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_{source}_2017')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ACLED

# COMMAND ----------

with open(f'{dirpath}/survey/acled_lcvr_ref_panda.pickle', 'rb') as handle:
    x_acled = pickle.load(handle)

# COMMAND ----------

source = 'acled'
# declare schema, change to pyspark df, save
schema = StructType([StructField("lat", FloatType(), True), StructField("lon", FloatType(), True), StructField("time", DateType(), True), StructField('event_count', FloatType(), True), StructField('fatal_count', FloatType(), True)])
x_acled = spark.createDataFrame(x_acled, schema)
x_acled.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_{source}_2017')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### y's

# COMMAND ----------

with open(f'{dirpath}/survey/y_lcvr_ref_panda.pickle', 'rb') as handle:
    y = pickle.load(handle)

# COMMAND ----------

y['y0_bin'] = y['y0'].apply(lambda x: 1 if x >= 0.5 else 0)
y['ya_25_bin'] = y['ya_25'].apply(lambda x: 1 if x >= 0.5 else 0)
y['ya_50_bin'] = y['ya_50'].apply(lambda x: 1 if x >= 0.5 else 0)
y['ya_75_bin'] = y['ya_75'].apply(lambda x: 1 if x >= 0.5 else 0)

# COMMAND ----------

# declare schema, change to pyspark df, save
schema = StructType([StructField("time", DateType(), True), StructField("lat", FloatType(), True), StructField("lon", FloatType(), True), StructField('y0', FloatType(), True), StructField('y0_nw', FloatType(), True), StructField('ya_25', FloatType(), True), StructField('ya_50', FloatType(), True), StructField('ya_75', FloatType(), True), StructField('ya_50_nw', FloatType(), True), StructField('y0_bin', IntegerType(), True), StructField('ya_25_bin', IntegerType(), True), StructField('ya_50_bin', IntegerType(), True), StructField('ya_75_bin', IntegerType(), True)])
y = spark.createDataFrame(y, schema)
y.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_y_outcome_2017')

# COMMAND ----------

# x_geo

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Annual

# COMMAND ----------

lscn = spark.sql("SELECT * FROM myanmar.lcvr_ref_landscan_2017")
lcvr = spark.sql("SELECT * FROM myanmar.lcvr_ref_landcover_2017")
print(lscn.count())
print(lcvr.count())

# COMMAND ----------

m = lcvr.join(lscn, on=['lat', 'lon'], how='inner')
print(m.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Monthly

# COMMAND ----------

fldas = spark.sql("SELECT * FROM myanmar.lcvr_ref_fldas_2017")
viirs = spark.sql("SELECT * FROM myanmar.lcvr_ref_viirs_2017")
print(fldas.count())
print(viirs.count())

# COMMAND ----------

m2 = fldas.join(viirs, on=['time', 'lat', 'lon'], how='inner')
print(m2.count())

# COMMAND ----------

acled = spark.sql("SELECT * FROM myanmar.lcvr_ref_acled_2017")
m2 = m2.join(acled, on=['time', 'lat', 'lon'], how='left').fillna(0, subset=['event_count', 'fatal_count'])
print(m2.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Annual, Monthly

# COMMAND ----------

m2 = m2.join(m, on=['lat', 'lon'])
print(m2.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Y

# COMMAND ----------

y = spark.sql("SELECT * FROM myanmar.lcvr_ref_y_outcome_2017")
m2 = m2.join(y, on=['time', 'lat', 'lon'], how='left')
print(m2.count())

# COMMAND ----------

tomodel = m2.where(col("y0_bin").isNotNull())
print(tomodel.count())

# COMMAND ----------

tomodel.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_model_2017')

# COMMAND ----------

toapply = m2.where(col("y0_bin").isNull())
print(toapply.count())

# COMMAND ----------

toapply.write.mode('append').format('delta').saveAsTable(f'myanmar.lcvr_ref_apply_2017')

# COMMAND ----------


