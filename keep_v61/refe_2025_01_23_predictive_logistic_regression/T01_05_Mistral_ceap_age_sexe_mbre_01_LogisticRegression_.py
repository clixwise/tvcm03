import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb

def inpu(file_path, filt_valu, filt_name):
   
    # ----
    # 1:Inpu
    # ----
    file_inpu = "../../inpu/InpuFile05.a.3a6_full.c4.UB.csv.oupu.csv"
    path_inpu = os.path.join(file_path, file_inpu)
    df1 = pd.read_csv(path_inpu, delimiter="|", na_filter=False, nrows=1400)

    #
    df2 = df1 if filt_valu is None else df1[df1[filt_name] == filt_valu]
    print(f"sexe : {filt_valu} : df1.size={len(df1)} df2.size={len(df2)}")
    #
    df11 = df2.copy() # keep all
    df12 = df2[~df2['ceap'].isin(['NA'])] # eliminate 'NA'
    df13 = df2[~df2['ceap'].isin(['NA', 'C0', 'C1', 'C2'])] # eliminate 'NA', 'C0', 'C1', 'C2'
    
    df_line = df11
    #df_tabl = df11.groupby(['name', 'doss', 'sexe', 'ceap']).agg({'age': 'mean'}).reset_index()

    
    trac = True
    if trac:
        print(f"\Input file filtered : df_line.size:{len(df_line)} df_line.type:{type(df_line)}\n{df_line}\n:{df_line.index}\n:{df_line.columns}")
        #print(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")
        #write(f"\nContingency table  : df_tabl.size:{len(df_tabl)} df_tabl.type:{type(df_tabl)}\n{df_tabl}\n:{df_tabl.index}")

    
    # ----
    # Exit
    # ----
    return df11, df12, df13

def sta1(df):
    print (">>>>")
    print ("sta1")
    print (">>>>")
    # Convert CEAP to numeric
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Prepare features and target
    X = df[['age', 'sexe', 'mbre']]
    y = df['ceap_numeric']

    # Encode categorical variables
    categorical_features = ['sexe', 'mbre']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical features with numerical features
    X_combined = pd.concat([X[['age']].reset_index(drop=True), X_encoded_df], axis=1)

    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_combined, y)

    # Preprocessing pipelines for both numeric and categorical data
    numeric_features = ['age']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', list(X_encoded_df.columns))])

    # Create preprocessing and modeling pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(multi_class='multinomial', max_iter=1000))])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nCohen's Kappa:\n", cohen_kappa_score(y_test, y_pred, weights='quadratic'))

    # Feature importance
    feature_importance = pd.DataFrame({'feature': model.named_steps['preprocessor'].get_feature_names_out(),
                                       'importance': np.mean(np.abs(model.named_steps['classifier'].coef_), axis=0)})
    print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))

def sta2(df):
    print (">>>>")
    print ("sta2")
    print (">>>>")
    # Convert CEAP to numeric
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Prepare features and target
    X = df[['age', 'sexe', 'mbre']]
    y = df['ceap_numeric']

    # Encode categorical variables
    categorical_features = ['sexe', 'mbre']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical features with numerical features
    X_combined = pd.concat([X[['age']].reset_index(drop=True), X_encoded_df], axis=1)

    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_combined, y)

    # Preprocessing pipelines for both numeric and categorical data
    numeric_features = ['age']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', list(X_encoded_df.columns))])

    # Create preprocessing and modeling pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train model
    grid_search.fit(X_train, y_train)

    # Evaluate model
    y_pred = grid_search.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nCohen's Kappa:\n", cohen_kappa_score(y_test, y_pred, weights='quadratic'))

    # Feature importance
    best_model = grid_search.best_estimator_.named_steps['classifier']
    feature_importance = pd.DataFrame({'feature': grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out(),
                                       'importance': best_model.feature_importances_})
    print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))

def sta3(df, use_polynomial=False):
    print (">>>>")
    print (f"sta3 : use_polynomial= {use_polynomial}")
    print (">>>>")
    # Convert CEAP to numeric
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Prepare features and target
    X = df[['age', 'sexe', 'mbre']]
    y = df['ceap_numeric']

    # Encode categorical variables
    categorical_features = ['sexe', 'mbre']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical features with numerical features
    X_combined = pd.concat([X[['age']].reset_index(drop=True), X_encoded_df], axis=1)

    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_combined, y)

    # Preprocessing pipelines for both numeric and categorical data
    numeric_features = ['age']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    if use_polynomial:
        numeric_transformer.steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', list(X_encoded_df.columns))])

    # Create preprocessing and modeling pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train model
    grid_search.fit(X_train, y_train)

    # Evaluate model
    y_pred = grid_search.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nCohen's Kappa:\n", cohen_kappa_score(y_test, y_pred, weights='quadratic'))

    # Feature importance
    best_model = grid_search.best_estimator_.named_steps['classifier']
    feature_importance = pd.DataFrame({'feature': grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out(),
                                       'importance': best_model.feature_importances_})
    print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))

# Example usage
# df = pd.read_csv('your_data.csv')
# stat(df, use_polynomial=False)  # Without polynomial features
# stat(df, use_polynomial=True)   # With polynomial features

# GBM
def stat_gbm(df, use_polynomial=False):
    
    print (">>>>")
    print (f"stat_gbm : use_polynomial= {use_polynomial}")
    print (">>>>")
    # Convert CEAP to numeric
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Prepare features and target
    X = df[['age', 'sexe', 'mbre']]
    y = df['ceap_numeric']

    # Encode categorical variables
    categorical_features = ['sexe', 'mbre']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical features with numerical features
    X_combined = pd.concat([X[['age']].reset_index(drop=True), X_encoded_df], axis=1)

    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_combined, y)

    # Preprocessing pipelines for both numeric and categorical data
    numeric_features = ['age']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    if use_polynomial:
        numeric_transformer.steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', list(X_encoded_df.columns))])

    # Create preprocessing and modeling pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', GradientBoostingClassifier(random_state=42))])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train model
    grid_search.fit(X_train, y_train)

    # Evaluate model
    y_pred = grid_search.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nCohen's Kappa:\n", cohen_kappa_score(y_test, y_pred, weights='quadratic'))

    # Feature importance
    best_model = grid_search.best_estimator_.named_steps['classifier']
    feature_importance = pd.DataFrame({'feature': grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out(),
                                       'importance': best_model.feature_importances_})
    print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))

# Example usage
# df = pd.read_csv('your_data.csv')
# stat_gbm(df, use_polynomial=False)  # Without polynomial features
# stat_gbm(df, use_polynomial=True)   # With polynomial features

def stat_xgboost(df, use_polynomial=False):
    
    print (">>>>")
    print (f"stat_xgboost : use_polynomial= {use_polynomial}")
    print (">>>>")
    # Convert CEAP to numeric
    df['ceap_numeric'] = pd.Categorical(df['ceap']).codes
    print(df['ceap_numeric'].unique())

    # Prepare features and target
    X = df[['age', 'sexe', 'mbre']]
    y = df['ceap_numeric']

    # Encode categorical variables
    categorical_features = ['sexe', 'mbre']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical features with numerical features
    X_combined = pd.concat([X[['age']].reset_index(drop=True), X_encoded_df], axis=1)

    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_combined, y)

    # Preprocessing pipelines for both numeric and categorical data
    numeric_features = ['age']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    if use_polynomial:
        numeric_transformer.steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', list(X_encoded_df.columns))])

    # Create preprocessing and modeling pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train model
    grid_search.fit(X_train, y_train)

    # Evaluate model
    y_pred = grid_search.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nCohen's Kappa:\n", cohen_kappa_score(y_test, y_pred, weights='quadratic'))

    # Feature importance
    best_model = grid_search.best_estimator_.named_steps['classifier']
    feature_importance = pd.DataFrame({'feature': grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out(),
                                       'importance': best_model.feature_importances_})
    print("\nFeature Importance:\n", feature_importance.sort_values('importance', ascending=False))


# Example usage
# df = pd.read_csv('your_data.csv')
# stat_xgboost(df, use_polynomial=False)  # Without polynomial features
# stat_xgboost(df, use_polynomial=True)   # With polynomial features
    
if __name__ == "__main__":

    # Step 1
    exit_code = 0           
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(__file__)
    print (f"len(sys.argv): {len(sys.argv)}")
    print (f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = script_dir
    
    # Step 2
    suppress_suffix = ".py"
    script_name = script_name[:-len(suppress_suffix)]
    jrnl_file_path = os.path.join(script_dir, f"{script_name}jrnl.txt")
    with open(jrnl_file_path, 'w') as file:
        
        # set_file_objc(file)
        
        # Step 21
        filt_valu = None
        filt_name = 'sexe'
        df11, df12, df13 = inpu(file_path, filt_valu, filt_name)
        print (df11)
        if True:
            sta1(df12)
            sta2(df12)
            sta3(df12, use_polynomial=False)
            sta3(df12, use_polynomial=True)
            stat_gbm(df12, use_polynomial=False)
            stat_gbm(df12, use_polynomial=True)
        if False : # creates error
            stat_xgboost(df12, use_polynomial=False)
            stat_xgboost(df12, use_polynomial=True)
        pass
    
'''

'''

