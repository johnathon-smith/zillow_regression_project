import numpy as np
import pandas as pd
import env
import wrangle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score

#The following function will create dummy variables, remove the original column, and return the new data frames
def get_dummy_vars(train, validate, test):
    train_dummies = pd.get_dummies(train[['county']], dummy_na=False, drop_first=True)
    train = pd.concat([train, train_dummies], axis = 1).drop(columns = ['county'])

    validate_dummies = pd.get_dummies(validate[['county']], dummy_na=False, drop_first=True)
    validate = pd.concat([validate, validate_dummies], axis = 1).drop(columns = ['county'])

    test_dummies = pd.get_dummies(test[['county']], dummy_na=False, drop_first=True)
    test = pd.concat([test, test_dummies], axis = 1).drop(columns = ['county'])

    return train, validate, test

#Now scale the X groups using the Robust Scaler
def scale_data(X_train, X_validate, X_test):
    #Create the scaler
    scaler = StandardScaler()

    #Fit the scaler on X_train
    scaler.fit(X_train)

    #Transform the data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validate_scaled, X_test_scaled

#Compare the new and original distributions just to be sure everything is okay
def compare_dists(train, cols_to_scale, cols_scaled):

    plt.figure(figsize=(18,6))

    for i, col in enumerate(cols_to_scale):
        i += 1
        plt.subplot(2,8,i)
        train[col].plot.hist()
        plt.title(col)

    for i, col in enumerate(cols_scaled):
        i += 9
        plt.subplot(2,8,i)
        train[col].plot.hist()
        plt.title(col)

    plt.tight_layout()
    plt.show()


def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)

#Get the baseline and print the RMSE
def get_baseline(y_train, y_validate, y_test, metric_df):
    #Change y_train and y_validate to be data frames so we can store the baseline values in them
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    #Calculate baseline based on mean
    baseline_mean_pred = y_train.tax_value.mean()
    y_train['baseline_mean_pred'] = baseline_mean_pred
    y_validate['baseline_mean_pred'] = baseline_mean_pred
    y_test['baseline_mean_pred'] = baseline_mean_pred

    #Calculate RMSE based on mean
    train_RMSE = mean_squared_error(y_train.tax_value, y_train['baseline_mean_pred']) ** .5
    validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate['baseline_mean_pred']) ** .5

    #Print RMSE
    print("RMSE using Mean\nTrain/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2),
        "\n")

    metric_df = make_metric_df(y_validate.tax_value, y_validate['baseline_mean_pred'], 'validate_baseline_mean', metric_df)

    return y_train, y_validate, y_test, metric_df

def get_ols_model(X_train_scaled, X_validate_scaled, y_train, y_validate, metric_df):
    #Create the model
    lm = LinearRegression(normalize = True)

    #Fit the model on scaled data
    lm.fit(X_train_scaled, y_train.tax_value)

    #Make predictions
    y_train['lm_preds'] = lm.predict(X_train_scaled)
    y_validate['lm_preds'] = lm.predict(X_validate_scaled)

    #Calculate the RMSE
    train_RMSE = mean_squared_error(y_train.tax_value, y_train['lm_preds']) ** .5
    validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate['lm_preds']) ** .5

    print("RMSE using OLS\nTrain/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

    metric_df = make_metric_df(y_validate.tax_value, y_validate['lm_preds'], 'validate_ols', metric_df)

    return lm, metric_df

def get_lars_models(X_train_scaled, X_validate_scaled, y_train, y_validate, metric_df):
    #Create a list to hold all the different models
    lars_models = []

    #Loop through different alpha values. Start with 1.
    for i in range(1, 21):
        #Create the model
        lars = LassoLars(alpha = i)
        
        #Fit the model
        lars.fit(X_train_scaled, y_train.tax_value)
        
        #Make predictions
        y_train[f'lars_alpha_{i}'] = lars.predict(X_train_scaled)
        y_validate[f'lars_alpha_{i}'] = lars.predict(X_validate_scaled)
        
        #Calculate RMSE
        train_RMSE = mean_squared_error(y_train.tax_value, y_train[f'lars_alpha_{i}']) ** .5
        validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate[f'lars_alpha_{i}']) ** .5

        #Add model to list of lars models
        lars_models.append({f'lars_alpha_{i}': lars})
        
        print(f'\nRMSE using LassoLars, alpha = {i}')
        print("Train/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

        metric_df = make_metric_df(y_validate.tax_value, y_validate[f'lars_alpha_{i}'], f'validate_lars_alpha_{i}', metric_df)

    return lars_models, metric_df

def get_glm_model(X_train_scaled, X_validate_scaled, y_train, y_validate, metric_df):
    #Create a list to hold all the models
    glm_models = []

    #Use a loop to try each power and several values for alpha
    for i in range(0, 3):

        #The following loop determines the alpha values
        for j in range(1, 11):
            #Create the model
            glm = TweedieRegressor(power = i, alpha = j)

            #Fit the model
            glm.fit(X_train_scaled, y_train.tax_value)

            #Make predictions
            y_train[f'glm_power_{i}_alpha_{j}_preds'] = glm.predict(X_train_scaled)
            y_validate[f'glm_power_{i}_alpha_{j}_preds'] = glm.predict(X_validate_scaled)

            #Calculate RMSE
            train_RMSE = mean_squared_error(y_train.tax_value, y_train[f'glm_power_{i}_alpha_{j}_preds']) ** .5
            validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate[f'glm_power_{i}_alpha_{j}_preds']) ** .5

            #Add model to the list
            glm_models.append({f'glm_{i}_{j}':glm})

            print(f'\nRMSE for Power = {i}, Alpha = {j}\n')
            print("Train/In-Sample: ", round(train_RMSE, 2), 
                "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

            metric_df = make_metric_df(y_validate.tax_value, y_validate[f'glm_power_{i}_alpha_{j}_preds'], f'validate_glm_power_{i}_alpha_{j}', metric_df)

    return glm_models, metric_df

def get_polynomial_model(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, metric_df):
    #Create a list to hold the models and a list to hold the different feature sets
    poly_models = []
    poly_test_features = []

    #Use a loop to try degrees 2 through 4
    for i in range(2,6):
        #Create new features
        pf = PolynomialFeatures(degree = i)

        #Fit to scaled data
        X_train_degree_i = pf.fit_transform(X_train_scaled)

        #Transform for validate and test as well
        X_validate_degree_i = pf.transform(X_validate_scaled)
        X_test_degree_i = pf.transform(X_test_scaled)

        #Now use linear regression with new features
        lm2 = LinearRegression(normalize = True)

        #Fit the model
        lm2.fit(X_train_degree_i, y_train.tax_value)

        #Make predicitons
        y_train[f'poly_preds_{i}'] = lm2.predict(X_train_degree_i)
        y_validate[f'poly_preds_{i}'] = lm2.predict(X_validate_degree_i)

        #Calculate RMSE
        train_RMSE = mean_squared_error(y_train.tax_value, y_train[f'poly_preds_{i}']) ** .5
        validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate[f'poly_preds_{i}']) ** .5

        #Add model to list
        poly_models.append({ f'poly_degree_{i}': lm2 })
        poly_test_features.append(X_test_degree_i)

        print(f'RMSE for Polynomial Features, degree = {i}')
        print("Train/In-Sample: ", round(train_RMSE, 2), 
            "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

        metric_df = make_metric_df(y_validate.tax_value, y_validate[f'poly_preds_{i}'], f'validate_poly_degree_{i}', metric_df)

    return poly_test_features, poly_models, metric_df

#The following function will plot the predictions of my best model.
def plot_predictions(y_test):
    plt.figure(figsize = (16, 8))
    plt.plot(y_test.tax_value, y_test.tax_value, color = 'green')
    plt.plot(y_test.tax_value, y_test['baseline_mean_pred'], color = 'red')
    plt.scatter(y_test.tax_value, y_test['poly_preds_3'], alpha = 0.5)
    plt.annotate('Ideal: Actual Tax Value', (1.0*10**6, 1.05*10**6), rotation = 25)
    plt.annotate('Baseline: Mean Tax Value', (1.15*10**6, .3*10**6), color = 'red')
    plt.title('My Best Model: Predictions')
    plt.xlabel('Actual Tax Value')
    plt.ylabel('Predicted Tax Value')
    plt.show()

#The following function will plot the residuals of my best model.
def plot_residuals(y_test):
    plt.figure(figsize=(16,8))
    plt.scatter( y_test.tax_value, y_test.tax_value - y_test['poly_preds_3'], alpha = .5)
    plt.axhline(label="No Error", color = 'green')
    plt.title('My Best Model: Residuals')
    plt.xlabel('Actual Tax Value')
    plt.ylabel('Prediction Error (Residuals)')
    plt.show()