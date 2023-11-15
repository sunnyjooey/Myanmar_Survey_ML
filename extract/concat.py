# Databricks notebook source
!pip install xarray

# COMMAND ----------

import pickle
import pandas as pd
import numpy as np
import xarray

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in data and merge

# COMMAND ----------

dirpath = '/dbfs/FileStore/Myanmar_Survey_ML/data/'

# COMMAND ----------

# open feature sets
with open(f'{dirpath}/geo/landscan/landscan2017_xarray.pickle', 'rb') as handle:
    lscn = pickle.load(handle)

with open(f'{dirpath}/geo/landcover/landcover2017_xarray.pickle', 'rb') as handle:
    lcvr = pickle.load(handle)

with open(f'{dirpath}/geo/viirs/2017/viirs2017_xarray.pickle', 'rb') as handle:
    viirs = pickle.load(handle)  

with open(f'{dirpath}/geo/fldas/2017/fldas2017_xarray.pickle', 'rb') as handle:
    fldas = pickle.load(handle)   

with open(f'{dirpath}/survey/acled_panda.pickle', 'rb') as handle:
    acled = pickle.load(handle)    

# COMMAND ----------

# open y 
with open(f'{dirpath}/survey/y_panda.pickle', 'rb') as handle:
    y = pickle.load(handle)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Last bit of harmonizing lat/lon

# COMMAND ----------

lscn = lscn.to_dataframe().reset_index().drop(['time','source'], axis=1)
lcvr = lcvr.to_dataframe().reset_index().drop(['time','source'], axis=1)

# COMMAND ----------

# check how much the lat/lons are off by
lscn_lat = lscn.lat.unique()
lcvr_lat = lcvr.lat.unique()
lscn_lat = lscn_lat[:len(lcvr_lat)]
print(max(lscn_lat-lcvr_lat))
print(min(lscn_lat-lcvr_lat))

lscn_lon = lscn.lon.unique()
lcvr_lon = lcvr.lon.unique()
lscn_lon = lscn_lon[:len(lcvr_lon)]
print(max(lscn_lon-lcvr_lon))
print(min(lscn_lon-lcvr_lon))

# create a dictionary for changing values
lat_dict = {lcvr_lat[i]: lscn_lat[i] for i in range(len(lscn_lat))}
lon_dict = {lcvr_lon[i]: lscn_lon[i] for i in range(len(lscn_lon))}

# COMMAND ----------

# convert landcover
lcvr['lat'] = lcvr['lat'].apply(lambda x: lat_dict[x] if x in lat_dict else np.nan)
lcvr['lon'] = lcvr['lon'].apply(lambda x: lon_dict[x] if x in lon_dict else np.nan)

# COMMAND ----------

# convert viirs
viirs = viirs.to_dataframe().reset_index().drop('source', axis=1)
viirs['lat'] = viirs['lat'].apply(lambda x: lat_dict[x] if x in lat_dict else np.nan)
viirs['lon'] = viirs['lon'].apply(lambda x: lon_dict[x] if x in lon_dict else np.nan)

# COMMAND ----------

# convert fldas
fldas = fldas.to_dataframe().reset_index().drop('source', axis=1)
fldas['lat'] = fldas['lat'].apply(lambda x: lat_dict[x] if x in lat_dict else np.nan)
fldas['lon'] = fldas['lon'].apply(lambda x: lon_dict[x] if x in lon_dict else np.nan)

# COMMAND ----------

print(lscn.shape[0])
print(lcvr.shape[0])
lscn = pd.merge(lscn, lcvr)
print(lscn.shape)
lscn.head()

# COMMAND ----------

print(lscn.shape[0])
print(viirs.shape[0])
lscn = pd.merge(lscn, viirs)
print(lscn.shape)
lscn.head()

# COMMAND ----------

print(lscn.shape[0])
print(fldas.shape[0])
lscn = pd.merge(lscn, fldas, on=['lat','lon','time'])
print(lscn.shape)
lscn.head()

# COMMAND ----------

print(lscn.shape[0])
print(acled.shape[0])
lscn = pd.merge(lscn, acled, how='left')
lscn = lscn.fillna({'event_count':0, 'fatal_count':0})
print(lscn.shape)
lscn.head()

# COMMAND ----------

print(lscn.shape[0])
print(y.shape[0])
lscn = pd.merge(lscn, y, how='left')
# DO NOT FILL NA!
print(lscn.shape)
lscn.head()
