# Databricks notebook source
import pandas as pd
import re
import mlflow

# COMMAND ----------

df = spark.sql('SELECT * FROM myanmar_ml.lcvr_ref_model_2017_fixed')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply models to the rest of the country

# COMMAND ----------

# get model runs and results
id = 389746355820867
max_results = 1000  # Maximum number of rows to retrieve
runs = mlflow.search_runs(experiment_ids=[id], max_results=max_results)

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

# fitting the model super manually - refitting on entire data, then applying
def run_apply_model(runs, run_name, df, app_df):
    # cols to drop
    drop = ['time', 'lat', 'lon', 'y0', 'y0_nw', 'ya_25', 'ya_50', 'ya_75', 'ya_50_nw', 'y0_bin', 'ya_25_bin', 'ya_50_bin', 'ya_75_bin', '202', '80', '140', '122', '220']
    
    # select supported columns - all columns
    from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
    supported_cols = [col for col in df.columns if col not in drop]
    col_selector = ColumnSelector(supported_cols)


    # boolean columns
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

    bool_imputers = []
    bool_pipeline = Pipeline(steps=[
        ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
        ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
        ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    bool_cols = [col for col in supported_cols if col.isdigit()]
    bool_transformers = [("boolean", bool_pipeline, bool_cols)]


    # numerical columns
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    num_imputers = []
    num_imputers.append(("impute_mean", SimpleImputer(), supported_cols))

    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ])
    numerical_transformers = [("numerical", numerical_pipeline, supported_cols)]

    from sklearn.compose import ColumnTransformer
    transformers = bool_transformers + numerical_transformers
    preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)


    # get parameter space from the model runs
    param = runs.loc[runs['tags.mlflow.runName']==run_name, 'params.classifier'].iloc[0]
    if not bool(re.match('TransformedTargetClassifier', param)):
        if bool(re.match('LGBMClassifier', param)):
            import lightgbm
            from lightgbm import LGBMClassifier
            param = param.strip(')').strip('LGBMClassifier(')
            param_str_lst = [elem.strip('\n').strip() for elem in param.split(',')]
            param_lst_lst = [elem.split('=') for elem in param_str_lst]
            space = {lst[0]:lst[1] for lst in param_lst_lst}
            space = {key: (float(val) if bool(re.search('\.', val)) else int(val)) for key,val in space.items()}
            lgbmc_classifier = LGBMClassifier(**space)
            
        elif bool(re.match('DecisionTreeClassifier', param)):
            from sklearn.tree import DecisionTreeClassifier
            param = param.strip(')').strip('DecisionTreeClassifier(')
            param_str_lst = [elem.strip('\n').strip() for elem in param.split(',')]
            param_lst_lst = [elem.split('=') for elem in param_str_lst]
            space = {lst[0]:lst[1] for lst in param_lst_lst}
            space = {key: (float(val) if bool(re.search('\.', val)) else int(val)) for key,val in space.items()}
            lgbmc_classifier = DecisionTreeClassifier(**space)

        elif bool(re.match('LogisticRegression', param)):
            from sklearn.linear_model import LogisticRegression
            param = param.strip(')').strip('LogisticRegression(')
            param_str_lst = [elem.strip('\n').strip() for elem in param.split(',')]
            param_lst_lst = [elem.split('=') for elem in param_str_lst]
            space = {lst[0]:lst[1] for lst in param_lst_lst}
            space = {key: (float(val) if bool(re.search('\.', val)) else int(val)) for key,val in space.items()}
            lgbmc_classifier = LogisticRegression(**space)

        elif bool(re.match('RandomForestClassifier', param)):
            from sklearn.ensemble import RandomForestClassifier
            param = param.strip(')').strip('RandomForestClassifier(')   
            param_str_lst = [elem.strip('\n').strip() for elem in param.split(',')]
            param_lst_lst = [elem.split('=') for elem in param_str_lst]
            space = {lst[0]:lst[1] for lst in param_lst_lst}
            space = {key: (float(val) if bool(re.search('\.', val)) else int(val)) for key,val in space.items()}
            lgbmc_classifier = RandomForestClassifier(**space)

        # data
        df = df.toPandas()

        # to get evaluation metrics
        print('Train - val - test model')
        from sklearn.model_selection import train_test_split
        
        split_train_df, split_test_df = train_test_split(df, test_size=0.4)
        split_test_df, split_val_df = train_test_split(split_test_df, test_size=0.5)
        
        # Separate target column from features and drop _automl_split_col_0000
        X_train = split_train_df.drop(target_col, axis=1)
        y_train = split_train_df[target_col]

        X_val = split_val_df.drop(target_col, axis=1)
        y_val = split_val_df[target_col]

        X_test = split_test_df.drop(target_col, axis=1)
        y_test = split_test_df[target_col]

        pipeline_val = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
        ])
        pipeline_val.fit(X_train, y_train)
        X_val_processed = pipeline_val.transform(X_val)

        model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("classifier", lgbmc_classifier),
        ])

        model.fit(X_train, y_train, classifier__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], classifier__eval_set=[(X_val_processed,y_val)])

        # Log metrics for the test set
        from mlflow.models import Model
        from mlflow import pyfunc
        from mlflow.pyfunc import PyFuncModel
        
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(target_col):y_test}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "test_" , "pos_label": 1 }
        )
        lgbmc_test_metrics = test_eval_result.metrics
        lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}
        test_run = pd.DataFrame({k:[val] for k,val in lgbmc_test_metrics.items()})
        one_run = runs.loc[runs['tags.mlflow.runName']==run_name, ['run_id', 'tags.mlflow.runName', 'metrics.test_f1_score', 'metrics.test_precision_score', 'metrics.test_recall_score', 'metrics.test_roc_auc']]
        one_run = one_run.reset_index().drop('index', axis=1)
        
        run = pd.concat([one_run, test_run], axis=1)
        
    else:
        run = None
    
    return run

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

def show_map(merge_data, one_run, mod):
    poor = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'prop_poor'].iloc[0], 2)
    auc = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_roc_auc'].iloc[0], 2)
    rec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_recall_score'].iloc[0], 2)
    prec = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_precision_score'].iloc[0], 2)
    f1 = round(one_run.loc[one_run['tags.mlflow.runName']==mod,'metrics.test_f1_score'].iloc[0], 2)
    title = f'{mod}\n poor {poor}; roc-auc {auc}; recall {rec}; precision {prec}; f1 {f1}'
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

# this function isn't working because X_val is needed
def get_shap():
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=824288948).fillna(mode)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=824288948).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

shp = gpd.read_file('/dbfs/FileStore/Myanmar_Survey_ML/data/geo/adm3_shapefile/mmr_polbnda_adm3_250k_mimu_1.shp')

# COMMAND ----------

mods = ['calm-sponge-868', 'omniscient-kit-369', 'bald-snipe-799', 'kindly-vole-791']

def show_model(runs, run_name, df, app_df, coords, shp):
    # remodeled and applied
    pred_remod, one_run_remod = run_apply_model(runs, run_name, df, app_df)
    # just applied
    pred_apply, one_run_apply = apply_model(runs, run_name, app_df)

    # create data for visualization
    app_df[f'{run_name}-manual'] = pred_remod
    app_df[run_name] = pred_apply
    viz = pd.concat([coords, app_df], axis=1)

    # geometry points
    geometry = [Point(xy)  for xy in zip(viz['lon'], viz['lat'])]
    viz_gdf = gpd.GeoDataFrame(viz, crs=shp.crs, geometry=geometry)

    # Project PrioGrid and Admin1 to Mercator
    viz_gdf = viz_gdf.to_crs(epsg=3857)
    shp = shp.to_crs(epsg=3857)
    
    # join
    viz_merge = gpd.tools.sjoin_nearest(viz_gdf, shp, how='inner')
    mn = viz_merge[['TS', f'{run_name}-manual', run_name]].groupby('TS').mean()
    mg = pd.merge(shp, mn, on='TS', how='outer')

    show_map(mg, one_run_remod, f'{run_name}-manual')
    show_map(mg, one_run_apply, run_name)
    show_graphs(runs, run_name)

# COMMAND ----------

run_name = 'calm-sponge-868'
show_model(runs, run_name, df, app_df, coords, shp)

# COMMAND ----------

run_name = 'kindly-vole-791'
show_model(runs, run_name, df, app_df, coords, shp)

# COMMAND ----------



# COMMAND ----------



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

# COMMAND ----------


