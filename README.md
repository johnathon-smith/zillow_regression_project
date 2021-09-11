## Zillow Regression Project
***

### Project Summary
***

#### Goals
* Predict the values of single unit properties that the tax district assesses using the property data from those with a transaction during the "hot months" (in terms of real estate demand) of May-August, 2017.
* Determine the state and county location of each property.
* Calculate and chart the tax rate distributions for each county.

#### Audience
* The Zillow data science team.

#### Deliverables
* A five minute verbal presentation supported with slides.
* A Github Repository containing:
    1) A clearly labeled final report jupyter notebook.
    2) The .py files necessary to reproduce my work for anyone with their own env.py file.
* Finally, a README.md file documenting my project planning with instructions on how someone could clone and reproduce my project on their own machine. Goals for the project, a data dictionary, and key findings and takeaways should be included.

#### Context
* The Zillow data I'm using was acquired from the Codeup Database.

#### Data Dictionary (Relevant Columns Only)
| Target | Datatype | Definition |
|:-------|:---------|:------------|
| tax_value | int | The assessed value of the property |

| Feature | Datatype | Definition |
|:--------|:---------|:------------|
| bedroom_count | int | The number of bedrooms in the property |
| bathroom_count | float | The number of bathrooms in the property (Includes values for half baths and other combinations) |
| home_area | int | The area of the property in square feet |
| county | str | The name of the county the property resides in |
| tax_amount | float | The amount of tax paid (Used to calculate county_tax_rate) |
| state | str | The name of the state the property resides in |
| county_tax_rate | float | The tax rate applied to the property (Calculated using tax_amount and tax_value) |

#### Initial Hypotheses

__Hypothesis 1__
H_0: bedroom_count is not linearly correlated with tax_value
H_a: bedroom_count is linearly correlated with tax_value
alpha = 0.05
Outcome: To be determined

__Hypothesis 2__
H_0: bathroom_count is not linearly correlated with tax_value
H_a: bathroom_count is linearly correlated with tax_value
alpha = 0.05
Outcome: To be determined

__Hypothesis 3__
H_0: home_area is not linearly correlated with tax_value
H_a: home_area is linearly correlated with tax_value
alpha = 0.05
Outcome: To be determined

***

### Executive Summary - Conclusions and Next Steps
***

* All properties are located in California.
* All properties exist in either Los Angelas, Orange, or Ventura county.

***

### My Process
***

##### Plan
- [x] Write a README.md file that details my process, my findings, and instructions on how to recreate my project.
- [x] Acquire the zillow data from the Codeup Database
- [x] Clean and prepare the zillow data:
    * Select only the useful columns
    * Remove or impute null values
    * Rename columns as necessary
    * Change data types as necessary
    * Calculate county_tax_rate
    * Create county and state columns
    * Remove entries that don't make sense or are illegal
    * Remove outliers
- [x] Plot individual variable distributions
- [x] Plot county_tax_rate distributions ( A Project Goal )
- [ ] Determine at least two initial hypotheses, run the statistical tests needed, evaluate the outcome, and record results.
- [ ] Split the data sets into X and y groups and scale the X groups before use in the model.
- [ ] Set baseline using tax_value mean or median.
- [ ] Create and evaluate model on train and validate sets
- [ ] Choose best model and evaluate it on test data set
- [ ] Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

##### Plan -> Acquire / Prepare
* Create and store functions needed to acquire and prepare the Zillow data from the Codeup Database in a wrangle.py file.
* The final function will return a pandas DataFrame.
* Import the wrangle.py module and use it to acquire the data in the Final Report Notebook.
* Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, ...).
* Plot distributions of individual variables.
* Plot distributions of county_tax_rates.
* List key takeaways.

___

##### Plan -> Acquire / Prepare -> Explore
* Create visuals that will help discover new relationships between features and the target variable.
* Test initial hypotheses and record results.
* List key takeaways.

___

##### Plan -> Acquire / Prepare -> Explore -> Model / Evaluate
* Split data into X and y groups.
* Scale the X groups.
* Set a baseline using tax_value mean or median.
* Create and evaluate at least three models on the train and validate data sets.
* Choose best model and evaluate it on the test data set.
* Document conclusions and next steps.

***

### Reproduce My Project

***

- [x] Read this README.md
- [ ] Download the wrangle.py, explore.py, and model.py modules into your working directory.
- [ ] Add your own env.py file to your directory. (user, password, host)
- [ ] Run the final report Jupyter notebook.