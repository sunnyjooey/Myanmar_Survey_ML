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
# MAGIC ##### ACLED

# COMMAND ----------

with open(f'{dirpath}/survey/acled_lcvr_ref_panda.pickle', 'rb') as handle:
    x_acled = pickle.load(handle)

# COMMAND ----------

source = 'acled'
# declare schema, change to pyspark df, save
schema = StructType([StructField("lat", FloatType(), True), StructField("lon", FloatType(), True), StructField("time", DateType(), True), StructField('event_count', FloatType(), True), StructField('fatal_count', FloatType(), True)])
x_acled = spark.createDataFrame(x_acled, schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### y's

# COMMAND ----------

with open(f'{dirpath}/survey/y_lcvr_ref_panda.pickle', 'rb') as handle:
    y = pickle.load(handle)

# COMMAND ----------

y['y0_bin'] = y['y0'].apply(lambda x: 1 if x > 0 else 0)
y['ya_25_bin'] = y['ya_25'].apply(lambda x: 1 if x > 0 else 0)
y['ya_50_bin'] = y['ya_50'].apply(lambda x: 1 if x > 0 else 0)
y['ya_75_bin'] = y['ya_75'].apply(lambda x: 1 if x > 0 else 0)

# COMMAND ----------

# declare schema, change to pyspark df, save
schema = StructType([StructField("time", DateType(), True), StructField("lat", FloatType(), True), StructField("lon", FloatType(), True), StructField('y0', FloatType(), True), StructField('y0_nw', FloatType(), True), StructField('ya_25', FloatType(), True), StructField('ya_50', FloatType(), True), StructField('ya_75', FloatType(), True), StructField('ya_50_nw', FloatType(), True), StructField('y0_bin', IntegerType(), True), StructField('ya_25_bin', IntegerType(), True), StructField('ya_50_bin', IntegerType(), True), StructField('ya_75_bin', IntegerType(), True)])
y = spark.createDataFrame(y, schema)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Annual

# COMMAND ----------

l2 = spark.sql("SELECT * FROM myanmar_ml.lcvr_ref_lscn_lcvr_2017")
print(l2.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Monthly

# COMMAND ----------

fldas = spark.sql("SELECT * FROM myanmar_ml.lcvr_ref_fldas_2017")
viirs = spark.sql("SELECT * FROM myanmar_ml.lcvr_ref_viirs_2017")
print(fldas.count())
print(viirs.count())

# COMMAND ----------

m2 = fldas.join(viirs, on=['time', 'lat', 'lon'], how='inner')
print(m2.count())

# COMMAND ----------

m2 = m2.join(x_acled, on=['time', 'lat', 'lon'], how='left').fillna(0, subset=['event_count', 'fatal_count'])
print(m2.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Annual, Monthly

# COMMAND ----------

m2 = m2.join(l2, on=['lat', 'lon'])
print(m2.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Y

# COMMAND ----------

m2 = m2.join(y, on=['time', 'lat', 'lon'], how='left')
print(m2.count())

# COMMAND ----------

tomodel = m2.where(col("y0_bin").isNotNull())
print(tomodel.count())

# COMMAND ----------

tomodel.write.mode('append').format('delta').saveAsTable(f'myanmar_ml.lcvr_ref_model_2017_fixed')

# COMMAND ----------

toapply = m2.where(col("y0_bin").isNull())
print(toapply.count())

# COMMAND ----------

toapply.write.mode('append').format('delta').saveAsTable(f'myanmar_ml.lcvr_ref_apply_2017')

# COMMAND ----------


