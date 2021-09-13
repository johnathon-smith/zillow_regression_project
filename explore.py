import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

#The following function plots all continuous variables and shows the regression line
def get_pairwise_charts(df):
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()

#Create a function that produces 3 different graphs for each
#combination of categorical and numerical columns
def plot_categorical_and_continuous_vars(df, cat_cols, num_cols):
    vars_to_chart = list(product(cat_cols, num_cols))
    
    for pair in vars_to_chart:
        #Fit the three related graphs in one row
        sns.set(rc={"figure.figsize":(10, 6)}) #width=10, height=6
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        plot1 = sns.stripplot(x=pair[0], y=pair[1], ax = ax1, data=df)
        plot1.set_xticklabels(plot1.get_xticklabels(), rotation = 45)
        
        plot2 = sns.boxplot(x=pair[0], y=pair[1], ax = ax2, data=df)
        plot2.set_xticklabels(plot2.get_xticklabels(), rotation = 45)
        
        plot3 = sns.violinplot(x=pair[0], y=pair[1], ax = ax3, data=df)
        plot3.set_xticklabels(plot3.get_xticklabels(), rotation = 45)
        
        plt.tight_layout()
        plt.show()


#The following function will compare p to the established alpha value, and print whether the nully hypothesis was rejected or not.
def is_significant(p, alpha):
    if p < alpha:
        print('p is less than alpha, so we reject the null hypothesis.')
    else:
        print('p is not less than alpha, so we fail to reject the null hypothesis.')