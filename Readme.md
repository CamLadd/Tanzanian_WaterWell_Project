Readme
# Overview
This project is based on the "Pump it Up: Data Mining the Water Table" hosted by DrivenData. In the notebooks provided in the repo of this project, you will find python code that entails the data cleaning and organization process, feature engineering, and the building of several machine learning models. Gridsearch will be used to optimize parameters of our models.

# Data Source
The data for this project comes from the Taarifa waterpoints dashboard, which aggregates data from the Tanzania Ministry of Water.
The Ministry of Water describes Taarifa as such "Taarifa is an open source platform for the crowd sourced reporting and triaging of infrastructure related issues. Think of it as a bug tracker for the real world which helps to engage citizens with their local government. We are currently working on an Innovation Project in Tanzania, with various partners."

The Taarifa homepage can be found by following this link: https://taarifa.org/

# Dataset Description and Observations
The data provided by DriveData was split into a 'Test Set' and a 'Train Set'. For this project, we used the 'Train Set', because it had much more data than that of the 'Test Set'. We considered merging thet two DataFrames, however we decided against it because the 'Train Set' has over 59k entries, which is more than enough data to work with. 
The data set has 41 columns, with each column seemingly representing different information. Each entry pertains to a specific well with a specific ID number, which we set as the index of the DataFrame for potential identification purposes. The columns in this data include the funder of the well, the water quality of the well, the water source, the basin that the well is located in, and more

Out of the 16 columns that we narrowed our DataFrame down to, we find that 8 are worth discussing:

## Region
Region can be a good indicator of well status, because how many functional vs non-functional wells differ a lot from region to region. For example, if you were to pick a well at random in the region 'Iringa', you would most likely find a functional well, but if you picked a well at random in 'Mtwara', you would most likely find a non-functional well!
## Basin
There are some basins that are clearly more successful when establishing wells. Two regions that stick out are 'Ruvuma/Southern Coast' and 'Lake Rukwa', because both of those basins seem to have a majority non-functional wells!
## Construction Year
When viewing the visualization, we can clearly identify that wells built in more recent years have a much higher chance of being functional.
## Quantity
It is clear that wells with 'enough' water quantity have a much higher chance of being functional, while almost ALL wells with a 'dry' quantity are non-functional.
## Waterpoint Type
The majority of wells with a waterpoint type of 'communal standpoint multiple' were non-functional, while the majority of wells with a waterpoint type of 'communal standpoint' were functional. This is an important difference we wish to express.
Overall, handpumps and communal standpoints (and to a lesser extent, improved spring) show a majority occurence of functional wells, while the rest show a majority occurence of non-functional wells
## Extraction Type
Overall, the data of wells that use a gravity and handpump extraction type have the highest occurence of functional wells, while motorpump and other extraction type are most likely to be non-functional in comparison. Submersible also has a majority functional, however it is not as pronounced as handpump and gravity.
## Source
Springs, rainwater harvesting, shallow wells, rivers, DBH machines, and hand boreholes (hand dtw) are most likely to be functional in comparison to the other source types. Noticeably, wells with their source as a river are functional significantly more often than if the source is a lake. This information was not conveyed in the 'source_type' column.
## Quality
Water of 'good' quality results in mostly functional wells, while those classified as an 'unknown' quality are mostly non-functional. They are unknown potentially due to the well being out of commission for such a long time that the water quality was no longer known by the time of recording.

## Data Cleaning and Organization
For our project, we dropped the columns:

['scheme_name', 'recorded_by', 'wpt_name', 'extraction_type', 'extraction_type_group',
           'region_code', 'district_code', 'lga', 'ward', 'public_meeting', 'date_recorded', 
           'source_type', 'source_class', 'waterpoint_type', 'water_quality', 'management_group', 
           'payment', 'quantity_group','subvillage', 'num_private', 'scheme_management', 'amount_tsh', 'latitude', 'longitude']

These columns were dropped due to redundancy and lack of meaningful data. When these columns are dropped, we have a total of 16 columns left, which is a good amount for a machine learning model as it is not nearly as noisy as the original 41 column DataFrame. After dropping these columns, we were left with much less missing values than before we dropped them, and after dropping the entries that had missing data, we only lost about 5% of the total data, which is something we can work with.

### Columns and their organization:
- Funder
	We replaced all of the values that were labeled as '0' with the label 'unknown' for readability and interpretability
	We kept the top 20 categories in the column, and everything outside of the top 20 was labeled as 'other'
- Installer 
	We kept the top 20 categories in the column, and everything outside of the top 20 was labeled as 'other'
- Population
	We replaced all values of the population that were 0 (which made little sense) with the mean of the population data
	We binned the values:
		<100
		100-200
		200-300
		300-400
		400-500
		>500
- Permit
	We labeled permit to 1 for True and 0 for False
- gps_height
	We binned the values:
		['-90m - sea level', 'sea level to 46m', '46m to 393m', '393m to 1017m', '1017m to 1316m', '1316m to 1586.75m', '1586.75m to 2770m']

# Modeling

When scoring our model, we are less focused on the precision of the model as a whole, but rather on the specific scores for recall and precision depending on class. While we still want higher precision for the model, we believe that the evaluation of these class specific metrics are more important because they help predict the specific problem in a more practical manner. 

For class 0, non-functional, recall is the most important metric. Recall is used when there is a high cost associated with a false negative. In this case, a false negative would indicate that we predicted that a truly non-functional well actually needs repair, or is completely functional. If we assume that these predictions are being made in order to find out where to allocate resources to bring water to an area, predicting that a well works or just needs repair could result in not enough resources being allocated to the area, if any at all. This could effectively deprive an area of a functioning well!

For class 1, functional, precision is the most important metric. Precision is used when there is a high cost associated with a false positive. In this case, a false positive would mean that we predict that a well that is truly non-functional or needs repair is functional. If we assume once again that these predicitons are being evaluated to make a decision on where to allocate resources, this could be detrimental because we may decide not to allocate resources to a well that actually needs it because we think it is fully functional when it actually isn't! If we have low precision for class 1, we could be leaving many areas without functional wells without even knowing it.

For class 2, needs repair, recall is the most important metric. In this case a false negative would indicate that we predicted that a well that actually needs repair is functional or not-functional. It would not be a big deal to falsely predict that a well needs repair, because if resources are allocated to that area and they find that the well actually works, then hooray, they don't need to use those resources. It also would not be a big deal to falsely predict that the well needs repair when it actually is non-functional, because when the area is surveyed, this will hopefully be noticed and the necessary resources will be allocated! 
However, it is detrimental to predict that a well is functional when it actually needs repair, because then we would be depriving the area of water, because there is little reason to spend the time to survey the area if it is predicted that the well is functional. It is also detrimental to predict that a well is non-functional if it actually needs repair, because then too many resources may be allocated (this is assuming that it takes less resources to repair a well than to build it from scratch). So false negatives are very costly here, and therefore recall is most important

## Analysis

When first evaluating all three models with default settings and a random seed of 123, we found that DecisionTree performed the least favorably, XGBoost performed the most favorably, and RandomForest sits comfortably in the middle. Using this information we decided to focus on RandomForest and XGBoost exclusively for our GridSearch. Using the parameter grids described above, we narrowed our hyperparameters down to specific values for each model:

RandomForest: (criterion='entropy', max_depth=15, n_estimators=150, random_state=123)

XGBoost: (learning_rate=0.05, max_depth=35, random_state=123, objective = 'multi:softprob', num_class=3)

Overall, XGBoost remained our most favorable model, with recall for class 0 and 2 being the highest out of all the models while also trying to maximize precision for class 1. The model as a whole had an accuracy of 79%, the highest accuracy out of any of our models.

# Conclusion
In conclusion, the best model based on our research is XGBoost. We chose this based on accuracy score, recall score for classses 0 and 2, and precision score for class 1. Our overall accuracy was 79%, significantly higher than our baseline accuracy of 54.3%!

For further plans, it would be beneficial to continue to run gridsearches and more models in order to try and maximize our scores of interest. We would need more time for this due to the computation time of running the models, as well as to account for the potential feature engineering and data manipulation that could be necessary depending on the models' results.