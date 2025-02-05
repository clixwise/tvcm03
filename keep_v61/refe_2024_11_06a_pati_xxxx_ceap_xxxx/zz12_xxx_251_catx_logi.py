import numpy as np
import pandas as pd
from util_file_mngr import write
import statsmodels.api as sm

# -------------------------------
# Logit Regression Test of Independence
# -------------------------------

def catx_logi(what, df, indx_cate_list, colu_cate_list, indx_name, colu_name, indx_name_stra, colu_name_ordi, df11):
    
    # Trac
    trac = True

    # 'df_tabl' only 
    if df11 is None:
        print(f"\nData : {what}\n(df_table) : Logit : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        write(f"\nData : {what}\n(df_table) : Logit : requires 'df_line' wich is None : {indx_name}:{indx_cate_list} : {colu_name}:{colu_cate_list}")
        return
    
    # Prec
    indx_cate_nam1 = indx_cate_list[0]
    indx_cate_nam2 = indx_cate_list[1]

    # Step 1 : create df1 with those age_bins where neither M nor F are 0
    dt = df.T
    df1 = dt[~((dt[indx_cate_nam1] == 0) | (dt[indx_cate_nam2] == 0))]
    df2 = dt[(dt[indx_cate_nam1] == 0) | (dt[indx_cate_nam2] == 0)]
    print(f"\nStep 1 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    print(f"\nStep 1 : df1.size:{len(df1)} df1.type:{type(df1)}\n{df1}\n:{df1.index}\n:{df1.columns}")
    print(f"\nStep 1 : df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")

    # Step 2 : eliminate from df11 all 'age_bin' listed in df2
    print(f"\nStep 1 : df11.size:{len(df11)} df2.type:{type(df11)}\n{df11}\n:{df11.index}\n:{df11.columns}")
    age_bins_to_exclude = df2.index
    df3 = df11[~df11[colu_name].isin(age_bins_to_exclude)] # colu_name='age_bin'
    df4 = df3[[colu_name, indx_name_stra]] # colu_name='age_bin' ; indx_name_stra='sexe_stra'
    print(f"\nStep 1 : df4.size:{len(df4)} df4.type:{type(df4)}\n{df4}\n:{df4.index}\n:{df4.columns}")

    # Step 3 : create age_bin columns
    age_bins_to_include = df1.index
    df5 = df4.copy()
    for col in age_bins_to_include:
        df5[col] = 0
    print(f"\nStep 1 : df5.size:{len(df5)} df5.type:{type(df5)}\n{df5}\n:{df5.index}\n:{df5.columns}")
    # Set the appropriate column to 1 based on the value in the age_bin column
    for index, row in df5.iterrows():
        age_bin_value = row[colu_name] # colu_name='age_bin'
        col_name = f'{age_bin_value}'
        if col_name in age_bins_to_include:
            df5.at[index, col_name] = 1
    print(f"\nStep 1 : df5.size:{len(df5)} df5.type:{type(df5)}\n{df5}\n:{df5.index}\n:{df5.columns}")

    # Step 4 : Define the dependent variable and independent variables
    X = df5.drop(columns=[colu_name,indx_name_stra]) # colu_name='age_bin' ; indx_name_stra='sexe_stra'
    y = df5[indx_name_stra] # indx_name_stra='sexe_stra'
    print(f"\nStep 4 : X.size:{len(X)} X.type:{type(X)}\n{X}\n:{X.index}")
    print(f"\nStep 4 : y.size:{len(y)} y.type:{type(y)}\n{y}\n:{y.index}")
    write(f"\nStep 4 : X.size:{len(X)} X.type:{type(X)}\n{X}\n:{X.index}")
    write(f"\nStep 4 : y.size:{len(y)} y.type:{type(y)}\n{y}\n:{y.index}")
    
    opti = False
    if opti:
        # Check for NaN values
        print("Missing values in X:", X.isna().sum().sum())
        print("Missing values in y:", y.isna().sum().sum())
        # Check for infinity values
        print("Infinite values in X:", np.isinf(X).sum().sum())
        print("Infinite values in y:", np.isinf(y).sum().sum())
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        # Assuming `X` is your design matrix (without the constant term if you've added it already)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif_data)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = X_scaled
    
    plot = False
    if plot: 
        # Plot 1   
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        sns.heatmap(X.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap of Predictors")
        plt.show()
        
        # Plot 2
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Bar plot of VIF values
        plt.figure(figsize=(10, 6))
        sns.barplot(x="VIF", y="Feature", data=vif_data, palette="viridis")
        plt.title("Variance Inflation Factor (VIF) for Each Predictor")
        plt.show()
        
        # Plot 3
        y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.xlabel("Gender")
        plt.ylabel("Frequency")
        plt.title("Distribution of Target Variable (Gender)")
        plt.show()
        
        # Plot 4
        # Fit model (using unstandardized or standardized X depending on the earlier analysis)
        logit_model = sm.Logit(y, sm.add_constant(X))
        result = logit_model.fit()

        # Standardized residuals
        residuals = result.resid_pearson

        plt.figure(figsize=(10, 6))
        plt.plot(residuals, 'o', markersize=4)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Observation")
        plt.ylabel("Standardized Residual")
        plt.title("Standardized Residual Plot")
        plt.show()
    
    # Step 5 : execute
    # Add a constant to the model for the intercept
    X = sm.add_constant(X)

    # Fit the Logistic Regression model
    model = sm.Logit(y, X) # X (predictor variables) ; y (dependent variable).
    result = model.fit(maxiter=100)

    # Step 6 : result 1
    # Print the summary of the logistic regression model : 'result' is the fitted logistic regression model
    print(f"\nData : {what}\nLogit :\n{result.summary()}")
    write(f"\nData : {what}\nLogit :\n{result.summary()}")
    
    # Step 7 : result 2
    # Extract the summary table as a DataFrame
    df_resu = result.summary2().tables[1]  # Access the coefficients table
    # Add a column for hypothesis decision based on p-value
    alph = 0.05
    df_resu['H'] = df_resu['P>|z|'].apply( lambda p: "Ha" if p < alph else "H0")

    frmt = lambda value: f"{value:.3e}" if value > 999 else (f"{value:.3e}" if value < 0.001 else f"{value:.3f}")
    df_resu['Coef.'] = df_resu['Coef.'].apply(frmt)
    df_resu['Std.Err.'] = df_resu['Std.Err.'].apply(frmt)
    df_resu['z'] = df_resu['z'].apply(frmt)
    df_resu['[0.025'] = df_resu['[0.025'].apply(frmt)
    df_resu['0.975]'] = df_resu['0.975]'].apply(frmt)

    # Add H0 and Ha columns
    H0 = f"H0 : No association between '{indx_name}' and '{colu_name}' (coef EQ 0)"
    Ha = f"Ha : Association between '{indx_name}' and '{colu_name}' (coef NE 0)"
    Hx = f"({colu_cate_list}) vs ({indx_cate_list})"

    print(f"\nData : {what}\nLogit regression :\n{H0}\n{Ha}")
    write(f"\nData : {what}\nLogit regression :\n{H0}\n{Ha}")
    with pd.option_context('display.width', None, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None): 
        print(f"\n{df_resu}")
        write(f"\n{df_resu}")
    print(f"{Hx}")
    write(f"{Hx}")
    pass