# Databricks notebook source
!pip install numpy==1.23.0
!pip install xarray
!pip install rioxarray

# COMMAND ----------

# restart the python kernel to import xarray correctly!
dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import datetime as dt
import pickle

import xarray as xr
import rioxarray
import rasterio

# COMMAND ----------

# paths
dirpath = '/dbfs/FileStore/Myanmar_Survey_ML/data'

# COMMAND ----------

with open(f'{dirpath}/survey/y_lcvr_ref_panda.pickle', 'rb') as handle:
    y = pickle.load(handle)

# COMMAND ----------

with open(f'{dirpath}/geo/all_x_lcvr_ref_xarray.pickle', 'rb') as handle:
    x_geo = pickle.load(handle)

# COMMAND ----------

with open(f'{dirpath}/survey/acled_lcvr_ref_panda.pickle', 'rb') as handle:
    x_acled = pickle.load(handle)
