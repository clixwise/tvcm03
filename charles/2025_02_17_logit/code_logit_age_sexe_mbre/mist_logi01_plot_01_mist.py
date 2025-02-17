import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from scipy.stats import norm

from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from util_file_mngr import write
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import scipy.stats as scs

from scipy.interpolate import make_interp_spline

def mist_logi01_plot_01_mist_a(what, df_line):      
    
    # Prec
    # ----
    df = df_line[['age','sexe', 'ceap']]
    
    # Data
    # ----
    df['sexe_nume'] = df['sexe'].map({'M': 0, 'F': 1}) # 'sexe' must be 'numeric'
    df['C3'] = (df['ceap'] == 'C3').astype(int) # Create binary outcome for C3
    df = df[['sexe_nume', 'age','C3', 'sexe', 'ceap']]
    print (df)
    
    # Exec
    # ----
    # Prepare the data for logistic regression
    X = pd.get_dummies(df[['sexe_nume', 'age']], drop_first=True)
    y = df['C3'] 
    log_model = sm.Logit(y, sm.add_constant(X)) # Fit the logistic regression model
    result = log_model.fit()
    print(result.summary())
    y_pred_prob = result.predict(sm.add_constant(X))
    
    # Mistral 1
    # =========

    # Extract coefficients
    confidence_intervals = result.conf_int()
    print (confidence_intervals)
    coefficients_df = pd.DataFrame({
        'Coefficient': result.params,
        'Std Error': result.bse,
        'z-value': result.tvalues,
        'P>|z|': result.pvalues,
        '95% CI Lower': confidence_intervals[0],
        '95% CI Upper': confidence_intervals[1]
    })
    if False:
        coefficients = [-1.1540, 0.3350, 0.0006]
        std_errors = [0.301, 0.154, 0.005]
        z_scores = [-3.836, 2.181, 0.124]
        p_values = [0.000, 0.029, 0.902]
        confidence_intervals = [
            (-1.744, -0.564),
            (0.034, 0.636),
            (-0.009, 0.010)
        ]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(coefficients_df.index, coefficients_df['Coefficient'], color='skyblue', label='Coefficient')
    ax.errorbar(coefficients_df.index, coefficients_df['Coefficient'],
                yerr=[coefficients_df['Coefficient'] - coefficients_df['95% CI Lower'],
                    coefficients_df['95% CI Upper'] - coefficients_df['Coefficient']],
                fmt='o', color='red', capsize=5)
    for i, p in enumerate(result.pvalues):
        if p <= 0.001:
            sig_symbol = '***'
        elif p <= 0.01:
            sig_symbol = '**'
        elif p <= 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = ''
        ax.text(i, coefficients_df['95% CI Upper'].iloc[i] + 0.05, sig_symbol, ha='center', color='black')

    # Add labels and title
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Logistic Regression Coefficients with 95% CI')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Show plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    ### 1. **Receiver Operating Characteristic (ROC) Curve**
    # The ROC curve is a graphical representation of the diagnostic ability of a binary classifier system. It plots the true positive rate (sensitivity) against the false positive rate (1-specificity) at various threshold settings.

        # Predict probabilities
    y_pred_prob = result.predict(sm.add_constant(X))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


### 2. **Confusion Matrix**
# A confusion matrix is a table used to describe the performance of a classification model. It shows the true vs predicted classifications.




    # Predict classes
    y_pred = result.predict(sm.add_constant(X))
    y_pred = (y_pred > 0.5).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-C3', 'C3'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


    ### 3. **Predicted Probabilities Plot**
    # A plot of predicted probabilities can help visualize the distribution of predicted values and how well the model separates the classes.



    # Plot predicted probabilities
    sns.histplot(y_pred_prob, bins=20, kde=True)
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.show()

    ### 4. **Residuals Plot**
    # Residuals plots can help identify patterns in the errors made by the model, which might indicate issues such as non-linearity or heteroscedasticity.


    # Calculate residuals
    residuals = y - y_pred_prob

    # Plot residuals
    sns.scatterplot(x=y_pred_prob, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Residuals')
    plt.show()

    ### 5. **Marginal Effects Plot**
    if True:
    # Marginal effects plots show how changes in the predictor variables affect the predicted probabilities, holding other variables constant.

        # Create a range of values for age
        age_range = np.linspace(df['age'].min(), df['age'].max(), 100)

        # Calculate predicted probabilities for each sexe_nume
        X_new = pd.DataFrame({'sexe_nume': [0] * len(age_range) + [1] * len(age_range),
                            'age': list(age_range) * 2})

        # Add a constant term (if your original model included it)
        X_new = sm.add_constant(X_new)

        # Split into male and female datasets
        X_new_male = X_new[X_new['sexe_nume'] == 0]
        X_new_female = X_new[X_new['sexe_nume'] == 1]

        # Make predictions
        probabilities_male = result.predict(X_new_male)
        probabilities_female = result.predict(X_new_female)

        # Plot marginal effects
        plt.plot(age_range, probabilities_male, label='Male')
        plt.plot(age_range, probabilities_female, label='Female')
        plt.title('Marginal Effects of Age on Predicted Probability')
        plt.xlabel('Age')
        plt.ylabel('Predicted Probability')
        plt.legend()
        plt.show()

    # Perplexity 1
    # ==========
    '''
    Yes, this logistic regression result can be graphed in Python. There are several ways to visualize the results:

    1. Coefficient plot: You can create a bar plot of the coefficients to show their relative importance[4]. This would display the magnitude and direction of the effect for 'sexe_nume' and 'age' on the probability of C3.
    2. Logistic regression curve: You can plot the logistic regression curve showing the relationship between the predictor variables and the probability of C3[1][3]. This would be particularly useful for visualizing the effect of 'age' on the probability of C3, while potentially using different colors or shapes to represent 'sexe_nume'.
    3. Decision boundary plot: For a 2D visualization, you can create a scatter plot of 'age' vs 'sexe_nume', with points colored by their C3 classification, and overlay the decision boundary[2].
    Citations:
    [1] https://www.statology.org/plot-logistic-regression-in-python/
    [2] https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
    [3] https://ajaytech.co/python-logistic-regression/
    [4] https://www.reddit.com/r/learnpython/comments/7ppvdd/best_way_to_plot_and_rank_logistic_regression/
    [5] https://realpython.com/logistic-regression-python/
    [6] https://stackoverflow.com/error
    [7] https://www.youtube.com/watch?v=GAiMnImkIZM
    [8] https://campus.datacamp.com/courses/machine-learning-for-time-series-data-in-python/validating-and-inspecting-time-series-models?ex=4
    '''
    # Plot 1 : Coefficients
    # ------
    '''
    Coefficient plot: You can create a bar plot of the coefficients to show their relative importance4. 
    This would display the magnitude and direction of the effect for 'sexe_nume' and 'age' on the probability of C3.
    '''
    if False:
        coef = result.params[1:]  # Exclude the intercept
        names = X.columns

        plt.figure(figsize=(10, 6))
        plt.bar(names, coef)
        plt.title('Logistic Regression Coefficients')
        plt.xlabel('Variables')
        plt.ylabel('Coefficient Value')
        plt.show()
        pass

    # Plot 2 : logistic regression curve
    # ------
    '''
    Certainly! To plot the logistic regression curve, we'll create a visualization that shows how the probability of C3 changes with age, 
    and we'll use different lines for males and females.
    This script does the following:

    1. Extracts the coefficients from the logistic regression model.
    2. Creates a range of ages from the minimum to the maximum in the dataset.
    3. Calculates the probabilities of C3 for males and females across the age range.
    4. Plots the logistic curves for males and females.
    5. Adds scatter plots of the actual data points.
    6. Adds labels, title, legend, and grid for better readability.

    This visualization will show:
    - Two curves: one for males and one for females, showing how the probability of C3 changes with age for each sex.
    - Scatter points of the actual data, allowing you to see how well the model fits the data.

    The difference between the male and female curves represents the effect of sex on the probability of C3, while the slope of the curves represents the effect of age. If the curves are nearly horizontal, it indicates that age has little effect on the probability of C3, which aligns with the non-significant p-value for age in your results.
    ''' 
    if False:
        # Extract coefficients
        intercept = result.params['const']
        age_coef = result.params['age']
        sex_coef = result.params['sexe_nume']

        # Create a range of ages
        ages = np.linspace(df['age'].min(), df['age'].max(), 100)

        # Calculate probabilities for males and females
        prob_male = expit(intercept + age_coef * ages)
        prob_female = expit(intercept + sex_coef + age_coef * ages)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(ages, prob_male, label='Male', color='blue')
        plt.plot(ages, prob_female, label='Female', color='red')

        # Add scatter plot of actual data
        plt.scatter(df[df['sexe'] == 'M']['age'], df[df['sexe'] == 'M']['C3'], 
                    color='blue', alpha=0.3, label='Male (actual)')
        plt.scatter(df[df['sexe'] == 'F']['age'], df[df['sexe'] == 'F']['C3'], 
                    color='red', alpha=0.3, label='Female (actual)')

        plt.xlabel('Age')
        plt.ylabel('Probability of C3')
        plt.title('Logistic Regression: Probability of C3 by Age and Sex')
        plt.legend()
        plt.grid(True)
        plt.show()
        pass

    # Plot 3 : detection boundary curve
    # ------
    '''
    Certainly! A decision boundary plot is a great way to visualize how the logistic regression model separates the two classes (C3 = 0 and C3 = 1) based on the predictor variables (age and sex). Here's a Python script to create this plot:

    This script does the following:

    1. Prepares the data for the logistic regression model.
    2. Fits a logistic regression model using scikit-learn.
    3. Creates a mesh grid covering the range of age and sex values.
    4. Predicts probabilities for each point in the grid.
    5. Creates a contour plot to visualize the decision boundary.
    6. Plots the actual data points, colored by their C3 classification.
    7. Adds labels, title, legend, and a colorbar for interpretation.

    In this plot:
    - The background color represents the probability of C3 = 1 as predicted by the model. Blue areas indicate low probability, while red areas indicate high probability.
    - The color gradient shows how the probability changes across different combinations of age and sex.
    - The actual data points are plotted as scatter points, with blue for C3 = 0 and red for C3 = 1.
    - The decision boundary is where the probability equals 0.5, which is typically where the color changes from blue to red.

    This visualization will help you see:
    - How age and sex jointly influence the probability of C3 classification.
    - Where the model draws the decision boundary between C3 = 0 and C3 = 1.
    - How well the model separates the two classes in the feature space.
    - Any patterns or clusters in the data that might not be apparent from the regression coefficients alone.
    '''
    if False:
        # Prepare the data
        X = df[['age', 'sexe_nume']]
        y = df['C3']

        # Fit a logistic regression model
        model = LogisticRegression()
        model.fit(X, y)

        # Create a mesh grid
        x0, x1 = X.iloc[:, 0], X.iloc[:, 1]
        xx, yy = np.mgrid[x0.min()-1:x0.max()+1:.01, x1.min()-1:x1.max()+1:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict the decision boundary
        probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the decision boundary
        contour = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=0.7)

        # Plot the data points
        ax.scatter(X[y==0]['age'], X[y==0]['sexe_nume'], c='blue', s=40, label='C3 = 0', alpha=0.5)
        ax.scatter(X[y==1]['age'], X[y==1]['sexe_nume'], c='red', s=40, label='C3 = 1', alpha=0.5)

        # Customize the plot
        ax.set_xlabel('Age')
        ax.set_ylabel('Sex (0=Male, 1=Female)')
        ax.set_title('Decision Boundary Plot: C3 Classification')
        ax.legend()
        plt.colorbar(contour)

        plt.show()
        pass
    
    # Perplexity 2
    # ============
    '''
    Yes, there are several other useful representations for logistic regression results that provide additional information:

    1. ROC Curve:
    The Receiver Operating Characteristic (ROC) curve visualizes the trade-off between true positive rate and false positive rate at various classification thresholds[5]. It's particularly useful for assessing the model's discriminative ability.
    2. Precision-Recall Curve:
    This curve shows the trade-off between precision and recall for different thresholds[5]. It's particularly useful for imbalanced datasets.
    3. Marginal Effects Plot:
    This visualization shows how the predicted probability changes as one predictor variable changes, while holding other variables constant[8]. 
    It's useful for understanding the impact of individual predictors.
    4. Coefficient Plot:
    A bar plot or forest plot of the model coefficients with their confidence intervals can provide insight into the relative importance 
    and statistical significance of predictors[2].
    5. Partial Dependence Plots:
    These plots show the marginal effect of one or two features on the predicted outcome of a machine learning model[5]. 
    They're useful for understanding complex relationships in the data.
    6. Confusion Matrix Heatmap:
    A visual representation of the confusion matrix using a heatmap can provide an intuitive understanding of the model's performance across different classes[5].

    ```python
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    ```

    These visualizations provide different perspectives on the model's performance and can help in interpreting the results more comprehensively[1][2][5][8].

    Citations:
    [1] https://ajaytech.co/python-logistic-regression/
    [2] https://static.bigml.com/static/html-doc/Classification_and_Regression/sec-visualizing-lr.html
    [3] https://www.statology.org/plot-logistic-regression-in-python/
    [4] https://library.virginia.edu/data/articles/visualizing-the-effects-of-logistic-regression
    [5] https://www.nobledesktop.com/learn/python/a-deep-dive-into-python-logistic-regression-theory-and-examples
    [6] https://www.youtube.com/watch?v=LURXqKWTQ4s
    [7] https://realpython.com/logistic-regression-python/
    [8] https://gagneurlab.github.io/dataviz/chap-log-reg.html
    [9] https://library.virginia.edu/data/articles/logistic-regression-four-ways-with-python
    '''
    
    # Plot 1 : ROC curve
    # ---
    '''
    The Receiver Operating Characteristic (ROC) curve visualizes the trade-off between true positive rate and false positive rate at various classification thresholds[5]. It's particularly useful for assessing the model's discriminative ability.
    
    You're right to ask about this, as the variables in the ROC curve example don't directly match your current setup. Let's adapt the ROC curve code to fit your case:

    1. First, we don't have separate test data, so we'll use the entire dataset.
    2. We need to calculate the predicted probabilities using our fitted model.

    Here's how you can modify the code to create an ROC curve for your logistic regression model:

    In this adapted version:

    1. `y` is your actual C3 values (0 or 1).
    2. `y_pred_proba` is calculated using the `predict` method of your fitted model. This gives the predicted probabilities for C3=1.
    3. We use these to calculate the ROC curve and AUC.

    This will give you an ROC curve that accurately represents the performance of your logistic regression model on your dataset.
    '''
    if False:
        # Calculate predicted probabilities
        y_pred_proba = result.predict(sm.add_constant(X))

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    # Plot 2
    # ------
    if True:
        # Calculate predicted probabilities
        y_pred_proba = result.predict(sm.add_constant(X))

        # Calculate precision and recall values
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)

        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        plt.show()

    '''
    To create a Precision-Recall Curve for your logistic regression model, you can use the following Python code:

    This code will:

    1. Calculate the predicted probabilities using your fitted model.
    2. Use `precision_recall_curve` from scikit-learn to compute precision and recall values at different thresholds.
    3. Plot the Precision-Recall curve.

    The resulting plot will show the trade-off between precision and recall for different classification thresholds[1][6]. A high area under the curve represents both high recall and high precision, indicating that the classifier is returning accurate results (high precision) as well as returning a majority of all positive results (high recall)[1].

    To interpret the curve:
    - The x-axis represents recall (true positive rate)
    - The y-axis represents precision (positive predictive value)
    - Each point on the curve corresponds to a different classification threshold
    - The curve typically starts at (1.0, y) and ends at (x, 0.0)[6]

    A model with perfect classification would have a precision of 1.0 for all recall values[1]. The closer the curve follows the top-right corner of the plot, the better the model's performance[6].

    This visualization is particularly useful for imbalanced datasets, as it focuses on the performance of the positive class (in your case, C3=1) without being influenced by true negatives[1][6].

    Citations:
    [1] https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    [2] https://research-information.bris.ac.uk/ws/portalfiles/portal/72164009/5867_precision_recall_gain_curves_pr_analysis_done_right.pdf
    [3] https://www.listendata.com/2019/07/precision-recall-curve-simplified.html
    [4] https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/
    [5] https://www.scikit-yb.org/en/latest/api/classifier/prcurve.html
    [6] https://www.datacamp.com/tutorial/precision-recall-curve-tutorial
    [7] https://www.machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    [8] https://www.blog.trainindata.com/precision-recall-curves/
    '''
    
    # Plot 3 : Marginal effectb plot
    # ------
    '''
    Certainly! A Marginal Effects Plot is an excellent way to visualize how changes in predictor variables affect the predicted probability of the outcome. For your logistic regression model with 'age' and 'sexe_nume' as predictors, we can create two marginal effects plots: one for age and one for sex. Here's how you can create these plots:

    This script does the following:

    1. Defines a function `marginal_effect` to calculate the marginal effect and its confidence interval for a given feature.

    2. For age:
    - Calculates the marginal effect across a range of age values.
    - Plots the marginal effect with a confidence interval.

    3. For sex:
    - Calculates the marginal effect for male and female.
    - Creates an error bar plot to show the marginal effect and its confidence interval for each sex.

    These plots will help you visualize:

    - How the probability of C3=1 changes as age increases, holding sex constant at its mean value.
    - The difference in probability of C3=1 between males and females, holding age constant at its mean value.

    The confidence intervals in these plots give you an idea of the uncertainty around these marginal effects. 
    If a confidence interval includes zero, it suggests that the marginal effect is not statistically significant at the 95% confidence level.
    
    '''
    if False:
        def marginal_effect(model, feature, X):
            # Calculate marginal effect
            mfx = model.get_margeff(at='mean')
            idx = list(X.columns).index(feature)
            me = mfx.margeff[idx]
            me_se = mfx.margeff_se[idx]
            
            # Calculate confidence interval
            z = norm.ppf(0.975)
            ci_lower = me - z * me_se
            ci_upper = me + z * me_se
            
            return me, ci_lower, ci_upper

        # Plot marginal effect for age
        age_range = np.linspace(X['age'].min(), X['age'].max(), 100)
        me_age = []
        ci_lower_age = []
        ci_upper_age = []

        for age in age_range:
            X_temp = X.copy()
            X_temp['age'] = age
            me, ci_l, ci_u = marginal_effect(result, 'age', X_temp)
            me_age.append(me)
            ci_lower_age.append(ci_l)
            ci_upper_age.append(ci_u)

        plt.figure(figsize=(10, 6))
        plt.plot(age_range, me_age, 'b-', label='Marginal effect')
        plt.fill_between(age_range, ci_lower_age, ci_upper_age, alpha=0.2, color='b')
        plt.xlabel('Age')
        plt.ylabel('Marginal effect on P(C3=1)')
        plt.title('Marginal Effect of Age on Probability of C3')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot marginal effect for sex
        sex_values = [0, 1]
        me_sex = []
        ci_lower_sex = []
        ci_upper_sex = []

        for sex in sex_values:
            X_temp = X.copy()
            X_temp['sexe_nume'] = sex
            me, ci_l, ci_u = marginal_effect(result, 'sexe_nume', X_temp)
            me_sex.append(me)
            ci_lower_sex.append(ci_l)
            ci_upper_sex.append(ci_u)

        # Convert lists to NumPy arrays
        me_sex = np.array(me_sex)
        ci_lower_sex = np.array(ci_lower_sex)
        ci_upper_sex = np.array(ci_upper_sex)

        plt.figure(figsize=(8, 6))
        plt.errorbar(sex_values, me_sex, yerr=[me_sex - ci_lower_sex, ci_upper_sex - me_sex], 
                    fmt='o', capsize=5, capthick=2, ecolor='b', markeredgecolor='b', markerfacecolor='w')
        plt.xlabel('Sex (0=Male, 1=Female)')
        plt.ylabel('Marginal effect on P(C3=1)')
        plt.title('Marginal Effect of Sex on Probability of C3')
        plt.xticks([0, 1], ['Male', 'Female'])
        plt.grid(True)
        plt.show()
    
    # Plot 4 : 4. Coefficient Plot
    # ------
    '''
    Certainly! A coefficient plot is an excellent way to visualize the magnitude, direction, and statistical significance of your logistic regression coefficients. Here's how you can create a coefficient plot with confidence intervals using your logistic regression results:

    This script does the following:

    1. Extracts coefficient names, values, and standard errors from your logistic regression results.
    2. Calculates 95% confidence intervals for each coefficient.
    3. Creates a plot where:
    - Coefficients are represented as blue dots.
    - Confidence intervals are shown as horizontal blue lines.
    - A vertical red dashed line at x=0 helps identify significant coefficients (those whose confidence intervals don't cross zero).
    4. Labels the y-axis with coefficient names and the x-axis with coefficient values.
    5. Adds text annotations showing the coefficient value and standard error for each predictor.

    This visualization will help you to:

    - Quickly identify which predictors have a significant effect (those whose confidence intervals don't cross zero).
    - Compare the relative importance of predictors (larger absolute values indicate stronger effects).
    - See the precision of your estimates (narrower confidence intervals indicate more precise estimates).

    Remember, in your case, you'll likely see two coefficients: one for 'sexe_nume' and one for 'age'. 
    This plot will help you visualize their relative importance and statistical significance in predicting the probability of C3 classification.
    '''
    if False:
        # Extract coefficient names, values, and standard errors
        coef_names = result.params.index[1:]  # Exclude the constant
        coef_values = result.params.values[1:]
        coef_errors = result.bse.values[1:]

        # Calculate confidence intervals (95%)
        conf_int = result.conf_int().values[1:]

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot coefficients as points
        plt.scatter(coef_values, range(len(coef_names)), color='blue', s=60)

        # Plot error bars (confidence intervals)
        plt.hlines(range(len(coef_names)), conf_int[:, 0], conf_int[:, 1], color='blue', lw=2)

        # Add vertical line at x=0
        plt.axvline(x=0, color='red', linestyle='--')

        # Customize the plot
        plt.yticks(range(len(coef_names)), coef_names)
        plt.xlabel('Coefficient Value')
        plt.title('Logistic Regression Coefficients with 95% Confidence Intervals')

        # Add coefficient values as text
        for i, (value, error) in enumerate(zip(coef_values, coef_errors)):
            plt.text(value, i, f' {value:.3f} (±{error:.3f})', va='center')

        plt.tight_layout()
        plt.show()

    # Plot 5
    # ----
    '''
    5. Partial Dependence Plots
    
    Certainly! Partial Dependence Plots (PDPs) are a great way to visualize the relationship between the target variable and a feature while accounting for the average effects of all other features. Here's how you can create PDPs for your logistic regression model:


    This script does the following:

    1. Prepares the data and fits a scikit-learn logistic regression model (which is needed for the `PartialDependenceDisplay` function).

    2. Creates two separate partial dependence plots:
    - One for Age, showing how the predicted probability of C3=1 changes with age, averaging over the effect of sex.
    - One for Sex, showing the difference in predicted probability between males and females, averaging over the effect of age.

    3. Creates a 2D partial dependence plot showing the joint effect of age and sex on the predicted probability of C3=1.

    These plots will help you visualize:

    - How the probability of C3=1 changes as age increases, accounting for the average effect of sex.
    - The difference in probability of C3=1 between males and females, accounting for the average effect of age.
    - The interaction between age and sex in predicting the probability of C3=1.

    The 2D plot is particularly useful for identifying any non-linear relationships or interactions between the two predictor variables.

    Note: We're using scikit-learn's LogisticRegression here instead of statsmodels because `PartialDependenceDisplay` is designed to work with scikit-learn models. 
    The results should be very similar to your statsmodels logistic regression.
    '''
    '''
    ERROR :
    The error you're encountering suggests that there's an issue with the shape of the 'sexe_nume' feature when trying to create the Partial Dependence Plot. This is likely because 'sexe_nume' is a binary variable (0 for Male, 1 for Female), and the PartialDependenceDisplay is expecting a continuous variable with more than two unique values[1][2].

    To resolve this issue, you can try the following modifications:

    1. For the 'sexe_nume' feature, use a different visualization method, such as a bar plot or a point plot, to show the partial dependence for the two categories.

    2. Alternatively, you can create a custom grid for 'sexe_nume' that includes intermediate values:

    ```python
    # Create a custom grid for 'sexe_nume'
    sex_grid = np.linspace(0, 1, num=100)

    # Use the custom grid in the PartialDependenceDisplay
    PartialDependenceDisplay.from_estimator(sk_model, X, ['sexe_nume'], ax=ax2, 
                                            grid_resolution=100, feature_grids=[sex_grid])
    ```

    3. If you want to keep the original binary nature of 'sexe_nume', you can use the `kind='individual'` parameter:

    ```python
    PartialDependenceDisplay.from_estimator(sk_model, X, ['sexe_nume'], ax=ax2, kind='individual')
    ```

    These modifications should help you visualize the partial dependence for the 'sexe_nume' feature without encountering the reshape error[6].

    Citations:
    [1] https://gemfury.com/alkaline-ml/python:scikit-learn/scikit_learn-0.23.1-pp36-pypy36_pp73-linux_x86_64.whl/content/inspection/_plot/partial_dependence.py
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html
    [3] https://stackoverflow.com/questions/61373393/plot-partial-dependence-from-scikit-learn-incorrectly-raises-notfittederror-fo
    [4] https://github.com/scikit-learn/scikit-learn/issues/30675
    [5] https://github.com/scikit-learn/scikit-learn/issues/25496
    [6] https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_partial_dependence_visualization_api.html
    [7] https://stackoverflow.com/questions/tagged/scikit_learn?tab=active&page=9
    [8] https://docs.esmvaltool.org/en/latest/_modules/esmvaltool/diag_scripts/mlr/models.html
    '''
    if False: # IN ERROR

        # Prepare the data
        X = df[['age', 'sexe_nume']]
        y = df['C3']

        # Fit a scikit-learn logistic regression model
        sk_model = LogisticRegression()
        sk_model.fit(X, y)

        # Create partial dependence plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # PDP for Age
        PartialDependenceDisplay.from_estimator(sk_model, X, ['age'], ax=ax1)
        ax1.set_ylabel('Partial Dependence')
        ax1.set_title('Partial Dependence of C3 on Age')

        # PDP for Sex
        # Create a custom grid for 'sexe_nume'
        sex_grid = np.linspace(0, 1, num=100)
        # Use the custom grid in the PartialDependenceDisplay
        PartialDependenceDisplay.from_estimator(sk_model, X, ['sexe_nume'], ax=ax2, 
                                                grid_resolution=100, feature_grids=[sex_grid])
        # ERROR : PartialDependenceDisplay.from_estimator(sk_model, X, ['sexe_nume'], ax=ax2)
        ax2.set_ylabel('Partial Dependence')
        ax2.set_title('Partial Dependence of C3 on Sex')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Male', 'Female'])

        plt.tight_layout()
        plt.show()

        # Create a 2D partial dependence plot
        fig, ax = plt.subplots(figsize=(10, 8))
        PartialDependenceDisplay.from_estimator(sk_model, X, ['age', 'sexe_nume'], ax=ax)
        ax.set_ylabel('Sex')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Male', 'Female'])
        ax.set_title('2D Partial Dependence of C3 on Age and Sex')

        plt.tight_layout()
        plt.show()
    
    # Plot 6 : Confusion matrxi
    # ----
    '''
     6. Confusion Matrix Heatmap:
    A visual representation of the confusion matrix using a heatmap can provide an intuitive understanding of the model's performance across different classes[5].
    
    The error you're encountering is due to the mismatch between the binary nature of your actual target variable (y) and the continuous probabilities output by the logistic regression model (y_pred_proba). To resolve this issue, you need to convert the predicted probabilities into binary predictions before creating the confusion matrix. Here's how you can modify your code:
    This modification does the following:

    1. We keep the calculation of `y_pred_proba` as before.
    2. We add a step to convert these probabilities to binary predictions using a threshold of 0.5. This is done with `(y_pred_proba > 0.5).astype(int)`.
    3. We use these binary predictions (`y_pred`) instead of the probabilities when creating the confusion matrix.

    This approach should resolve the "Classification metrics can't handle a mix of binary and continuous targets" error and allow you to create and visualize the confusion matrix correctly[1][2][3].

    Citations:
    [1] https://stackoverflow.com/questions/69875073/confusion-matrix-valueerror-classification-metrics-cant-handle-a-mix-of-binary/69875281
    [2] https://www.reddit.com/r/WGU_MSDA/comments/1ai10m2/d208_task_2_confusion_matrix/
    [3] https://datascience.stackexchange.com/questions/46019/continuous-variable-not-supported-in-confusion-matrix
    [4] https://medium.datadriveninvestor.com/building-a-confusion-matrix-from-scratch-85a8bfb97626?gi=8766908e4e31
    [5] https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    [6] https://www.kaggle.com/discussions/general/107836
    '''
    if False:
        # Calculate predicted probabilities
        y_pred_proba = result.predict(sm.add_constant(X))
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        # Create confusion matrix
        cm = confusion_matrix(y, y_pred) 

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    
    pass

def mist_logi01_plot_01_mist(what, df_logit):
    if True:
        pass
        mist_logi01_plot_01_mist_a("what", df_logit)

    pass