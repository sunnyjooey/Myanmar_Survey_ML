# Databricks notebook source
# script to explore & viz Myanmar survey data 
# --> basic initial explores
# --> check latitudinal / longitudinal distribution of surveys
# --> map surveys
# --> aggregate surveys to admin3 and plot totals, number poors, proportion poor per admin3

import pandas as pd
import numpy as np
import geopandas as gp
from shapely.geometry import Point


## get input data
# read in survey data
surveyDF = pd.read_stata('/dbfs/FileStore/Myanmar_Survey_ML/data/survey/Assets_household_level.dta')

# read in admin units
admin3MFilename='/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adminData_GeoBoundaries/geoBoundaries_MMR_ADM3.shp'
admin3DF=gp.read_file(admin3MFilename)


## basic characteristics of input data
# survey data
numSurveys=len(surveyDF.index)
numAttrPSurvey=len(list(surveyDF.columns.values.tolist()))

# notes from looking at the survey data: 
# -- the survey data contains a dichotomos variable poor/non-poor that appears to use the poverty_line 1302.951 as also given elsewhere, and it also contains a classification into different poverty groups.
# -- there are some columns which might be useful to consider down the line as predictors, as we could use these to explore correlations or develop a simple model and then get data on these predictors from elsewhere to extrapolate to other areas (e.g. electricity, source of light, roofs,). 


# get poverty baserate
poorBinDF=surveyDF['poor']
PropPoor=float(poorBinDF.sum()/poorBinDF.count())
display(PropPoor)

# num admin3 units
numAdmin3Units=len(admin3DF.index)


# COMMAND ----------

# explore spatial distribution of surveys
import matplotlib.pyplot as plt

# map surveys over admin3
surveyGeometry = [Point(xy)  for xy in zip(surveyDF['s0q23'], surveyDF['s0q22'])]
surveyLocGeoDF = gp.GeoDataFrame(surveyDF, crs=admin3DF.crs, geometry=surveyGeometry)
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DF.plot(ax=base,alpha=0.2, color='grey')
surveyLocGeoDF.plot(ax=base,markersize=1, color='blue', marker='o')



# COMMAND ----------

# get survey's lat/lon 
surveyLocations = np.array([[surveyDF.loc[i,'s0q22'], surveyDF.loc[i,'s0q23']] for i in surveyDF.index])

# plot directional distribution of lat-lon
latsSurveys=surveyLocations[:,0]
fig=plt.hist(latsSurveys)
plt.xlabel('latitude')
plt.ylabel('number of surveys')
plt.show()

lonsSurveys=surveyLocations[:,1]
plt.hist(lonsSurveys)
plt.xlabel('longitude')
plt.ylabel('number of surveys')
plt.show()
# note: the surveys are not equally distributed along lats and lons but centered around the central lat and lon; 

# note: there are a few surveys with lat/lon outside of Myanmar; should clean this at some point?


# COMMAND ----------

# aggregate surveys to admin3 via spatial join
MercatorProjCode=3857
WSG84CRSCode=4326
surveyLocGeoDF=surveyLocGeoDF.to_crs(epsg=MercatorProjCode)
admin3DF=admin3DF.to_crs(epsg=MercatorProjCode)

# aggregate grid-cell points to admin1 by nearest neighbour join
print('starting nearest neighbour join ...')
surveyPointsInAdmin3 = gp.tools.sjoin_nearest(surveyLocGeoDF,admin3DF,how='inner')
print('finished nearest neighbour join ...')

# project back to WSG84
surveyLocGeoDF=surveyLocGeoDF.to_crs(epsg=WSG84CRSCode)
admin3DF=admin3DF.to_crs(epsg=WSG84CRSCode)
surveyPointsInAdmin3=surveyPointsInAdmin3.to_crs(epsg=WSG84CRSCode)
# for checking: PsurveyPointsInAdmin3.plot()
       



# COMMAND ----------

# get survey stats per admin3
allAdmin3=admin3DF['shapeID']
targetCols=[
    'shapeID',
    'Admin3_name',
    'NumSurveys_PerAdmin3',
    'NumPoor_PerAdmin3',
    'PropPoor_PerAdmin3']
numTargetCols=len(targetCols)
numTargetRows=len(allAdmin3)
surveysPerAdmin3DF = pd.DataFrame(np.zeros((numTargetRows,numTargetCols)),columns=targetCols)
surveysPerAdmin3DF = surveysPerAdmin3DF.astype(
    {'shapeID': 'object', 
    'Admin3_name': 'object', 
    'NumSurveys_PerAdmin3': 'int',
    'NumPoor_PerAdmin3': 'int',
    'PropPoor_PerAdmin3': 'float'})

for iA in range(0,len(allAdmin3)):
    iAdmin3ID=allAdmin3[iA]
    iSurvPerAdmin3=surveyPointsInAdmin3[surveyPointsInAdmin3['shapeID']==iAdmin3ID]    
    iNumSurv=len(iSurvPerAdmin3.index)
    if iNumSurv>0:
        iAdmin3Name=iSurvPerAdmin3['shapeName'].iloc[0]
        iPoorPAdmin3=iSurvPerAdmin3['poor']
        iNumPoor=sum(iPoorPAdmin3)
        iPropPoor=iNumPoor/iNumSurv
    else:
        iAdmin3Name=''
        iNumSurv=np.nan
        iNumPoor=np.nan
        iPropPoor=np.nan  
    #display(iAdmin3Name)
    #display(iNumSurv,iNumPoor,iPropPoor)
    #display(iSurvPerAdmin3)
    surveysPerAdmin3DF['shapeID'].iloc[iA]=iAdmin3ID
    surveysPerAdmin3DF['Admin3_name'].iloc[iA]=iAdmin3Name
    surveysPerAdmin3DF['NumSurveys_PerAdmin3'].iloc[iA]=iNumSurv
    surveysPerAdmin3DF['NumPoor_PerAdmin3'].iloc[iA]=iNumPoor
    surveysPerAdmin3DF['PropPoor_PerAdmin3'].iloc[iA]=iPropPoor
    
# get number of admin3 without surveys
allAdmin3WithoutSurveys=surveysPerAdmin3DF[surveysPerAdmin3DF['NumSurveys_PerAdmin3']==0]
numAdmin3WithoutSurveys=len(allAdmin3WithoutSurveys)
display('Number of admin3 without surveys: '+str(numAdmin3WithoutSurveys))

# check distribution of surveys per admin3
plt.bar(range(0,numAdmin3Units),surveysPerAdmin3DF['NumSurveys_PerAdmin3'])
plt.title('Number of surveys per admin3')
plt.xlabel('admin3')
plt.xlabel('number of surveys per admin3')
plt.show()

meanNumSurveysPerAdmin3=np.mean(surveysPerAdmin3DF['NumSurveys_PerAdmin3'])
maxNumSurveysPerAdmin3=np.max(surveysPerAdmin3DF['NumSurveys_PerAdmin3'])
display('mean and max number surveys per admin3: '+str(meanNumSurveysPerAdmin3)+' '+str(maxNumSurveysPerAdmin3))

plt.hist(surveysPerAdmin3DF['NumSurveys_PerAdmin3'],50)
plt.title('Distribution of survey numbers per admin3')
plt.xlabel('number of surveys per admin3')
plt.ylabel('frequency of number of surveys')
plt.show()

plt.hist(surveysPerAdmin3DF['NumPoor_PerAdmin3'],50)
plt.title('Number of poor household per admin3')
plt.xlabel('number of poor households per admin3')
plt.ylabel('frequency of number of poor households')
plt.show()

plt.hist(surveysPerAdmin3DF['PropPoor_PerAdmin3'],50)
plt.title('Proportion of poor per admin3')
plt.xlabel('proportion of poor households per admin3')
plt.ylabel('frequency of proportion of poor households')
plt.show()



# COMMAND ----------

display(surveysPerAdmin3DF)

# COMMAND ----------

display(surveyDF)

# COMMAND ----------

# map surveys per admin3

admin3DFSurveys=admin3DF.merge(surveysPerAdmin3DF, on='shapeID')

# map number of surveys per admin3 
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DFSurveys.plot(ax=base,column='NumSurveys_PerAdmin3',legend=True,vmax=100,cmap='cividis').set_title('Number surveys')

# map number of poor per admin3 
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DFSurveys.plot(ax=base,column='NumPoor_PerAdmin3',legend=True,vmax=20,cmap='magma').set_title('Number poor households')

# map proportion of poor per admin3
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DFSurveys.plot(ax=base,column='PropPoor_PerAdmin3',legend=True,vmax=1.0,cmap='RdYlGn_r').set_title('Proportion poor')




# COMMAND ----------


