# Loan Status Prediction
## Goal of the project
Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), we can build a model that can predict whether or not a borrower will pay back their loan. This way, when we get a new potential customer, we can assess whether or not they are likely to repay the loan.

The "loan_status" column contains our label.

Note: This project is done as part of my coursework from Udemy (Python for Data Science and Machine Learning Bootcamp by Jose Portilla)

## Steps Followed:
* Data is obtained from Kaggle datasets. ( https://www.kaggle.com/wordsforthewise/lending-club )
* Initially basic info on the dataset is obtained and EDA is performed.
    + loan_amnt has an almost perfect correlation with the "installment" feature.
    + It looks like F and G subgrades don't get paid back that often.
* There are some missing values in the dataset.
    + emp_title - 22927
    + emp_length - 18301
    + title - 1755
    + revol_util - 276
    + mort_acc - 37795
    + pub_rec_bankruptcies - 35
  > Dropped both emp_title,emp_length due to many missing and large unique values.
  
  > The title column is simply a string subcategory/description of the purpose column. Hence dropped the title column.
  
  >  the total_acc feature correlates with the mort_acc. We will group the data frame by the total_acc and calculate the mean value for the mort_acc per total_acc entry.
  
  > revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Removing the rows that are missing those values in those columns with dropna().
  
* Categorical features are converted to numerical using dummy variables.
* Feature engineering:
    + Address column is converted to a zip code column by accessing the zip codes from the addresses.
    + earliest_cr_line - This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.
* Splitting the data using train test split and normalizing the data. Here I used minmaxscaler to normalize the data.
* Finally built a sequential model using TensorFlow.
