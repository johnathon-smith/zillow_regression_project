import numpy as np
import pandas as pd
import env
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#remove_outliers function, credit to John Salas from Codeup
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#Plot the individual distributions
def get_dists(df):
    for col in df.columns:
        sns.histplot(x = col, data = df)
        plt.title(col)
        plt.show()

#Will use in wrangle function. Changes fips value to associated county name.
def change_fips_to_county(value):
    if value == '6037':
        return 'Los Angelas'
    elif value == '6059':
        return 'Orange'
    else:
        return 'Ventura'

#The following function will acquire the zillow data from the Codeup Database and return a cleaned/prepared data frame.
def wrangle_zillow():
    #Set up mysql query
    zillow_query = """
    SELECT * FROM properties_2017
    LEFT JOIN predictions_2017 ON predictions_2017.parcelid = properties_2017.parcelid
    LEFT JOIN heatingorsystemtype ON heatingorsystemtype.heatingorsystemtypeid = properties_2017.heatingorsystemtypeid
    LEFT JOIN propertylandusetype ON propertylandusetype.propertylandusetypeid = properties_2017.propertylandusetypeid
    WHERE (properties_2017.propertylandusetypeid = 261
        OR properties_2017.propertylandusetypeid = 279)
        AND (predictions_2017.transactiondate >= '2017-05-01'
            AND predictions_2017.transactiondate <= '2017-08-31');
    """

    #Set up mysql url
    zillow_url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'

    #Acquire data
    zillow = pd.read_sql(zillow_query, zillow_url)

    #Begin preparing
    #Select only the needed columns
    zillow = zillow[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'fips', 'taxvaluedollarcnt', 'taxamount']]

    #rename the columns to be more understandable and easier to read
    zillow.rename(columns = {'bedroomcnt':'bedroom_count',
                    'bathroomcnt':'bathroom_count',
                    'calculatedfinishedsquarefeet':'home_area',
                    'taxvaluedollarcnt':'tax_value',
                    'taxamount':'tax_amount'}, inplace = True)

    #Fill missing 'home_area' values with 'home_area' median
    zillow.home_area = zillow.home_area.fillna(zillow.home_area.median())

    #Remove other na values
    zillow = zillow.dropna()

    #Now convert bedroom_count, home_area, and tax_value to ints
    zillow.bedroom_count = zillow.bedroom_count.astype(int)
    zillow.home_area = zillow.home_area.astype(int)
    zillow.tax_value = zillow.tax_value.astype(int)

    #Convert 'fips' to a string since it is categorical.
    zillow.fips = zillow.fips.astype(int)
    zillow.fips = zillow.fips.astype(str)

    #Change 'fips' to associated county names and rename column to 'county'
    zillow.fips = zillow.fips.apply(change_fips_to_county)
    zillow.rename(columns = {'fips':'county'}, inplace = True)

    #Add the county states to the df. They are all located in california
    zillow['state'] = 'California'

    #Now calculate tax_rate for each county using tax_value and tax_amount
    zillow['county_tax_rate'] = zillow.tax_amount / zillow.tax_value

    #Now remove things that don't make sense and/or are impossible/illegal.
    #If something doesn't sound like the average 'single family residential' property, drop it.
    zillow = zillow[(zillow.bedroom_count > 0) & (zillow.bathroom_count> 0)]
    zillow = zillow[zillow.bedroom_count <= 5]
    zillow = zillow[zillow.bathroom_count <= 3]
    zillow = zillow[zillow['home_area'] <= 5000]
    zillow = zillow[zillow['home_area'] >= (120 * zillow.bedroom_count)]
    zillow = zillow[zillow.tax_amount <= 20_000]

    #Now remove any outliers in the numerical columns
    cols_to_evaluate = zillow.select_dtypes('int').columns.append(zillow.select_dtypes('float').columns)
    zillow = remove_outliers(zillow, 2.5, cols_to_evaluate)

    return zillow

def train_validate_test_split(df, seed = 123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''

    #Drop unnecessary columns before splitting
    df.drop(columns = ['tax_amount', 'state', 'county', 'county_tax_rate'], inplace = True)

    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

#The following function will plot the tax-rate distributions for each county
def get_tax_rate_dists(zillow):
    #Change the size of the charts
    sns.set(rc={"figure.figsize":(16, 8)}) #width=16, height=8
    
    #Plot each county tax rate individually
    sns.histplot(data = zillow[zillow.county == 'Los Angelas'].county_tax_rate, label='Los Angelas')
    plt.title('Los Angelas County Tax Rates')
    plt.xlabel('County Tax Rates')
    plt.show()
    
    sns.histplot(data = zillow[zillow.county == 'Orange'].county_tax_rate, color = 'orange', label='Orange')
    plt.title('Orange County Tax Rates')
    plt.xlabel('County Tax Rates')
    plt.show()
    
    sns.histplot(data = zillow[zillow.county == 'Ventura'].county_tax_rate, color = 'green', label = 'Ventura')
    plt.title('Ventura County Tax Rates')
    plt.xlabel('County Tax Rates')
    plt.show()

    #Plot the county tax rates together for easy comparison
    sns.histplot(data = zillow[zillow.county == 'Los Angelas'].county_tax_rate, label = 'Los Angelas County')
    plt.xlabel('County Tax Rates')
    
    sns.histplot(data = zillow[zillow.county == 'Orange'].county_tax_rate, color = 'orange', label='Orange County')
    plt.xlabel('County Tax Rates')
    
    sns.histplot(data = zillow[zillow.county == 'Ventura'].county_tax_rate, color = 'green', label = 'Ventura County')
    plt.xlabel('County Tax Rates')

    plt.legend()
    plt.title('County Tax Rate Distributions')

    plt.show()