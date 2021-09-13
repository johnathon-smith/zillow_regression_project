import numpy as np
import pandas as pd
import env
import wrangle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error

#Now scale the X groups using the Robust Scaler
def scale_data(X_train, X_validate, X_test):
    #Create the scaler
    scaler = RobustScaler()

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
        plt.subplot(2,4,i)
        train[col].plot.hist()
        plt.title(col)

    for i, col in enumerate(cols_scaled):
        i += 5
        plt.subplot(2,4,i)
        train[col].plot.hist()
        plt.title(col)

    plt.tight_layout()
    plt.show()

#Get the baseline and print the RMSE
def get_baseline(y_train, y_validate):
    #Change y_train and y_validate to be data frames so we can store the baseline values in them
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    #Calculate baseline based on mean
    baseline_mean_pred = y_train.tax_value.mean()
    y_train['baseline_mean_pred'] = baseline_mean_pred
    y_validate['baseline_mean_pred'] = baseline_mean_pred

    #Calculate RMSE based on mean
    train_RMSE = mean_squared_error(y_train.tax_value, y_train['baseline_mean_pred']) ** .5
    validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate['baseline_mean_pred']) ** .5

    #Print RMSE
    print("RMSE using Mean\nTrain/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2),
        "\n")

    return y_train, y_validate

def get_ols_model(X_train_scaled, X_validate_scaled, y_train, y_validate):
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

    return lm

def get_lars_models(X_train_scaled, X_validate_scaled, y_train, y_validate):
    #Create a list to hold all the different models
    lars_models = []

    #Loop through different alpha values. Start with 1.
    for i in range(1, 101, 10):
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
        lars_models.append({f'lars+alpha_{i}': lars})
        
        print(f'\nRMSE using LassoLars, alpha = {i}')
        print("Train/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

    return lars_models

def get_glm_model(X_train_scaled, X_validate_scaled, y_train, y_validate):
    #Create the model
    glm = TweedieRegressor(power = 0, alpha = 1)

    #Fit the model
    glm.fit(X_train_scaled, y_train.tax_value)

    #Make predictions
    y_train['glm_preds'] = glm.predict(X_train_scaled)
    y_validate['glm_preds'] = glm.predict(X_validate_scaled)

    #Calculate RMSE
    train_RMSE = mean_squared_error(y_train.tax_value, y_train['glm_preds']) ** .5
    validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate['glm_preds']) ** .5

    print("RMSE using TweedieRegressor\nTrain/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

    return glm

def get_polynomial_model(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate):
    #Create new features
    pf = PolynomialFeatures(degree = 2)

    #Fit to scaled data
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    #Transform for validate and test as well
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)

    #Now use linear regression with new features
    lm2 = LinearRegression(normalize = True)

    #Fit the model
    lm2.fit(X_train_degree2, y_train.tax_value)

    #Make predicitons
    y_train['poly_preds'] = lm2.predict(X_train_degree2)
    y_validate['poly_preds'] = lm2.predict(X_validate_degree2)

    #Calculate RMSE
    train_RMSE = mean_squared_error(y_train.tax_value, y_train['poly_preds']) ** .5
    validate_RMSE = mean_squared_error(y_validate.tax_value, y_validate['poly_preds']) ** .5

    print("RMSE using Polynomial Features\nTrain/In-Sample: ", round(train_RMSE, 2), 
        "\nValidate/Out-of-Sample: ", round(validate_RMSE, 2))

    return X_test_degree2, lm2