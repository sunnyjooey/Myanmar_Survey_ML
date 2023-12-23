# Databricks notebook source
# explore naive/simple models as baseline 
#
# aim: ensure that complex models are better than naive assumptions
#
# approach: produce a set of naive/simple models and assess performance
# - naive 1: set all to poor
# - naive 2: set all to rich
# - naive 3: set 50/50
# - simple 1: random set to poor with prob. from baserate 
# - simple 2: random set to poor with prob from admin3 specific pov. rates 
#
# --------------------------------------------------

import pandas as pd
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

## get input data
# read in survey data
surveyDF = pd.read_stata('/dbfs/FileStore/Myanmar_Survey_ML/data/survey/Assets_household_level.dta')
numSurveys=len(surveyDF.index)
numAttrPSurvey=len(list(surveyDF.columns.values.tolist()))

# read in admin units
admin3MFilename='/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adminData_GeoBoundaries/geoBoundaries_MMR_ADM3.shp'
admin3DF=gp.read_file(admin3MFilename)

## separate surveys into training and test-data
# ...say 70% for training and 30% for testing
numTrainingData=int(0.7*numSurveys)
numTestData=numSurveys-numTrainingData
surveyIndices=surveyDF.index

trainingDataIndices=random.sample(range(int(min(surveyIndices)),int(max(surveyIndices))),numTrainingData)
TrainingData=surveyDF.loc[trainingDataIndices]

testDataIndices=np.zeros((numTestData))
k=0
for i in surveyIndices:
    itest=True
    for j in trainingDataIndices:
        if int(i)==int(j):
            #display(i)
            itest=False
    if itest:
        testDataIndices[k]=i
        k=k+1
TestData=surveyDF.loc[testDataIndices]

# for checking
sortedTrainingDataInd=np.sort(trainingDataIndices)
sortedTestDataInd=np.sort(testDataIndices)


# COMMAND ----------

# compute base-rate in training set
poorBinDF=TrainingData['poor']
PropPoorTrainingData=float(poorBinDF.sum()/poorBinDF.count())
display(PropPoorTrainingData)

# plot training data
surveyGeometry = [Point(xy)  for xy in zip(TrainingData['s0q23'], TrainingData['s0q22'])]
surveyLocGeoDF = gp.GeoDataFrame(TrainingData, crs=admin3DF.crs, geometry=surveyGeometry)
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DF.plot(ax=base,alpha=0.2, color='grey')
surveyLocGeoDF.plot(ax=base,markersize=1, color='blue', marker='o')


# COMMAND ----------

# aggregate to admin3
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
display(numAdmin3WithoutSurveys)

# plot distribution of surveys per admin3
plt.hist(surveysPerAdmin3DF['NumSurveys_PerAdmin3'],25)
plt.title('Number of surveys per admin3')
plt.show()

plt.hist(surveysPerAdmin3DF['NumPoor_PerAdmin3'],25)
plt.title('Number of poor household per admin3')
plt.show()

plt.hist(surveysPerAdmin3DF['PropPoor_PerAdmin3'],25)
plt.title('Proportion of poor per admin3')
plt.show()


# COMMAND ----------

# compute & plot poverty rate per admin3 in training set
admin3DFSurveys=admin3DF.merge(surveysPerAdmin3DF, on='shapeID')

# map number of surveys per admin3 
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DFSurveys.plot(ax=base,column='NumSurveys_PerAdmin3',legend=True,vmax=100,cmap='cividis').set_title('Number surveys')

# map number of poor per admin3 
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DFSurveys.plot(ax=base,column='NumPoor_PerAdmin3',legend=True,vmax=20,cmap='magma').set_title('Number poor households')

# map proportion of poor per admin3
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DFSurveys.plot(ax=base,column='PropPoor_PerAdmin3',legend=True,vmax=1.0,cmap='viridis').set_title('Proportion poor')



# COMMAND ----------

# produce model predictions & evaluate
#
# naive/simple models
# - naive 1: set all to poor
# - naive 2: set all to rich
# - naive 3: set 50/50
# - simple 1: random set to poor with prob. from baserate 
# - simple 2: random set to poor with prob from admin3 specific pov. rates 
#
# performance testing: 
# - recall: sensitivity; true positives / all actual positives
# - precision: positive predictive value; true positives / all predicted positives
# - roc-auc: area under roc curve // leave out for now
# - f1: harmonic mean of recall and precision; 2*(recall*precision)/(recall+precision)
#
# ---------------------------------------------

# get true values / test-data
true_test_values=TestData['poor']
print(true_test_values)
display(np.mean(true_test_values))




# COMMAND ----------


# - naive 1: set all to poor
# prediction
model_naive1_allPoor=np.ones(numTestData)

# evaluation
precision, recall, f1, counts =precision_recall_fscore_support(
    y_true=true_test_values, 
    y_pred=model_naive1_allPoor, 
    beta=1,
    average='binary',
    pos_label=1)

display('precision naive1 - all poor:'+str(precision))
display('recall naive1 - all poor:'+str(recall))
display('f1 naive1 - all poor:'+str(f1))

# COMMAND ----------


# - naive 2: set all to rich
model_naive2_allRich=np.zeros(numTestData)

# evaluation
precision, recall, f1, counts =precision_recall_fscore_support(
    y_true=true_test_values, 
    y_pred=model_naive2_allRich, 
    beta=1,
    average='binary',
    pos_label=1)

display('precision naive2 - all rich:'+str(precision))
display('recall naive2 - all rich:'+str(recall))
display('f1 naive2 - all rich:'+str(f1))


# COMMAND ----------


# - naive 3: set 50/50
model_naive3_fiftyfifty=np.empty(numTestData)
model_naive3_fiftyfifty[:]=np.nan
for i in range(0,numTestData):
    model_naive3_fiftyfifty[i]=random.randint(0,1)
# test=np.mean(model_naive3_fiftyfifty)

# evaluation
precision, recall, f1, counts =precision_recall_fscore_support(
    y_true=true_test_values, 
    y_pred=model_naive3_fiftyfifty, 
    beta=1,
    average='binary',
    pos_label=1)

display('precision naive3 - 50/50:'+str(precision))
display('recall naive3 - 50/50:'+str(recall))
display('f1 naive3 - 50/50:'+str(f1))

# COMMAND ----------

# - simple 1: random set to poor with prob. from baserate 
model_simple1_baseRate=np.empty(numTestData)
model_simple1_baseRate[:]=np.nan
for i in range(0,numTestData):
    iRand=random.uniform(0,1)
    if iRand<=PropPoorTrainingData:
        model_simple1_baseRate[i]=1
    else:
        model_simple1_baseRate[i]=0
#test=np.mean(model_simple1_baseRate)


# evaluation
precision, recall, f1, counts =precision_recall_fscore_support(
    y_true=true_test_values, 
    y_pred=model_simple1_baseRate, 
    beta=1,
    average='binary',
    pos_label=1)

display('precision simple1 - national baserate:'+str(precision))
display('recall simple1 - national baserate:'+str(recall))
display('f1 simple1 - national baserate:'+str(f1))

# COMMAND ----------

# - simple 2: random set to poor with prob from admin3 specific pov. rates
# --> need to determine admin3 for each test-data point

# plot test data
surveyGeometryTest = [Point(xy)  for xy in zip(TestData['s0q23'], TestData['s0q22'])]
surveyLocTestGeoDF = gp.GeoDataFrame(TestData, crs=admin3DF.crs, geometry=surveyGeometryTest)
base = admin3DF.boundary.plot(linewidth=0.7, edgecolor="black")
admin3DF.plot(ax=base,alpha=0.2, color='grey')
surveyLocTestGeoDF.plot(ax=base,markersize=1, color='blue', marker='o')

# aggregate test-data to admin3
MercatorProjCode=3857
WSG84CRSCode=4326
surveyLocTestGeoDF=surveyLocTestGeoDF.to_crs(epsg=MercatorProjCode)
admin3DF=admin3DF.to_crs(epsg=MercatorProjCode)

# aggregate test-data to admin3
print('starting nearest neighbour join ...')
surveyPointsTestDataInAdmin3 = gp.tools.sjoin_nearest(surveyLocTestGeoDF,admin3DF,how='inner')
print('finished nearest neighbour join ...')

# project back to WSG84
surveyLocTestGeoDF=surveyLocTestGeoDF.to_crs(epsg=WSG84CRSCode)
admin3DF=admin3DF.to_crs(epsg=WSG84CRSCode)
surveyPointsTestDataInAdmin3=surveyPointsTestDataInAdmin3.to_crs(epsg=WSG84CRSCode)

model_simple2_geoBaseRate=np.empty(numTestData)
model_simple2_geoBaseRate[:]=np.nan
for i in range(0,numTestData):
    
    # get test-data index
    iTestDataIndex=testDataIndices[0]
    #display(iTestDataIndex)
    
    # get admin3 of this test-data point
    iAdmin3=surveyPointsTestDataInAdmin3['shapeID'].iloc[int(iTestDataIndex)]
    #display(iAdmin3)

    # get poverty rate / proportion of poor in that admin3 in training data
    iTraining=admin3DFSurveys[admin3DFSurveys['shapeID']==iAdmin3]
    iPropPoorTraining=float(iTraining['PropPoor_PerAdmin3'])
    #display(iPropPoorTraining)

    # generate prediction depending on admin3 pov rate
    iRand=random.uniform(0,1) 
    if iRand<=iPropPoorTraining:
        model_simple2_geoBaseRate[i]=1
    else:
        model_simple2_geoBaseRate[i]=0
        
test=np.mean(model_simple2_geoBaseRate)
display(test)
#display(model_simple2_geoBaseRate)


# evaluation
precision, recall, f1, counts =precision_recall_fscore_support(
    y_true=true_test_values, 
    y_pred=model_simple2_geoBaseRate, 
    beta=1,
    average='binary',
    pos_label=1)

display('precision simple2 - admin3 baserate:'+str(precision))
display('recall simple2 - admin3 baserate:'+str(recall))
display('f1 simple2 - admin3 baserate:'+str(f1))



