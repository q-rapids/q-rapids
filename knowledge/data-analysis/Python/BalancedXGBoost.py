##################################################################################################
# PURPOSE:  Deriving XGBoost regression moodel from example data
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime (dt), time, sys, pickle, pandas (pd), numpy (np), scipy.stats,
#           collections.namedtuple, multiprocessing, itertools, matplotlib.pyplot (plt)
#           seaborn (sns),
#           imblearn.over_sampling.SMOTENC, imblearn.over_sampling.SMOTE, 
#           imblearn.over_sampling.RandomOverSampler, imblearn.under_sampling.RandomUnderSampler,
#           sklearn.model_selection.GridSearchCV, sklearn.model_selection.RandomizedSearchCV.
#           sklearn.model_selection.cross_validate, sklearn.model_selection. TimeSeriesSplit,
#           sklearn.model_selection.cross_val_score, sklearn.metrics.scorer.make_scorer
#           sklearn.metrics (mt), xgboost (xgb)
#           Please read and comply to the applicable usage licenses.
####################################################################################################

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## LOAD DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
input_path = os.path.normpath("<location/directorry of input data>")
output_path = os.path.normpath("<location/directorry of output data>")
data_folder = "PICKLE"

## Specify location and load data
data_set = "<file name>"
data_version = "<data version time stamp>"
data_variant = "<data variant>"

data_source = os.path.join(input_path, 'PICKLE',
                           data_set +
                           data_version +
                           data_variant + '.pickle')
df_source = pd.read_pickle(data_source)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SELECT DATA: Date, File, and Commit_ID are not necessary for the analysis
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Sort data on Date and FileName (to ensure correct time series split during analysis)
df_source.sort_values(by = ['Date', 'File'], ascending = True, inplace = True)

## Select relevant features
relevant_features = [
    'WeekDay',
    'Month',
    'Classes',
    'Code_Smells',
    'Cognitive_Cplx',
    'Comment_Lines',
    'Comment_Density',
    'Complexity',
    'Duplicate_Blocks',
    'Duplicate_Files',
    'Duplicate_Lines',
    'Duplicate_Lines Density',
    'Files',
    'Functions',
    'Total_LOC',
    'Non_Comment_LOC',
    'Reliability',
    'Security',
    'Statements',
    'BLOCKER',
    'CRITICAL',
    'INFO',
    'MAJOR',
    'MINOR',
    'Issue_Type_Bug'
]

df = df_source[relevant_features].copy()

## Select predictor and response features
response_feature_num = 'Issue_Type_Bug'
predictor_features = [feat for feat in relevant_features if feat != response_feature_num]
predictor_features_cat = df[predictor_features].select_dtypes(include=[np.object]).columns.tolist()
predictor_features_num = df[predictor_features].select_dtypes(include=[np.number]).columns.tolist()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HANDLE MISSING DATA: replace missing data on numeric columns with '0'
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##ToDo-OPTIONAL: Drop rows where Issue_Type_Bug is NaN
df = df.dropna(subset = [response_feature_num]).copy()
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_features] = df[num_features].fillna(value = 0)#, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## BALANCE Data
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Define column name for labeling mninority- and majority-class cases (observations)
balance_feature = 'Balance_Class'

## Add temporary attribute to code rare observations (in this case #Bug_Issues > 0)
def label_minority_cases(case, feature, lower_bound, upper_bound):
    if (case[feature] >= lower_bound) and (case[feature] <= upper_bound):
        return 'Majority' # majority case
    else:
        return 'Minority' # minority case

df[balance_feature] = df.apply(lambda case: label_minority_cases(case, response_feature_num, 0, 0), axis = 1)

## Visualize histogram of the response feature after discretization
plt = count_plot(data = df,
                 feature = balance_feature,
                 title = "Distribution of majority and majority clases)",
                 label = "Cases",
                 annotation = 'count')

## Select relevant featues (remove temporary feature for labeling minority/majority class)
explanatory_features = df.columns.to_list()
explanatory_features.remove(balance_feature)

## Split data for balacing
y = df[balance_feature].copy()
X = df[explanatory_features].copy()

## Remember column names and types (because balancing method will loose them)
cols_dict = {}
for column in X.columns:
        cols_dict.update({column:X[column].dtype})

## Set the number of cases to be oversampled
balance_count = np.count_nonzero(y == 'Majority')

## Oversample minority class up to number of majority-class observations
categorical_features = [X.columns.get_loc(c) for c in predictor_features_cat if c in X]

sampling_strategy = {'Minority': balance_count, 'Majority':balance_count}
smotenc = SMOTENC(categorical_features = categorical_features,
                  sampling_strategy = sampling_strategy,
                  k_neighbors = 5,
                  n_jobs = os.cpu_count() - 1)
X, y = smotenc.fit_resample(X, y)

## Reassign column names and types from original data frame
X = pd.DataFrame(data = X, columns = list(cols_dict.keys()))
X = X.astype(dtype = cols_dict)

## Visualize histogram of the response feature after discretization
plt = count_plot(data = pd.Series(y),
                 title = "Distribution of majority and majority clases)",
                 label = "Categories",
                 annotation = 'Count')

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SELECT TRAINING and TESTING Data Sets
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

explanatory_features = X.columns.to_list()
explanatory_features.remove(response_feature_num)
response_feature = response_feature_num

## Get data Frame
y = X[response_feature].copy()
X = X[explanatory_features].copy()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# PREPARE DATA: CATEGORICAL->NUMERICAL (Python tree methods cannot handle categorical features :-(
# => One-hot-encoding
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Select categorical features to encode
features_to_encode = [
                      'WeekDay',
                      'Month'
                     ]
df[features_to_encode].describe()

# Encode selected features ('get_dummies' returns the data frame with categorical features
# exchanged by one-hot encodings (adding encodigs and removing original features is not necessary)
X = pd.get_dummies(X, columns = features_to_encode, drop_first = False)

## Check correlation of explanatory features
corr_matrix = X.corr(method = 'spearman')
corr_heatmap (corr_matrix)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SEARCH FOR OPTIMAL HYPERPARAMETERS: GRID SEARCH
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#--------------------------------------------------------------------------------------------------
#  STEP 0: Prepare context
#--------------------------------------------------------------------------------------------------

##Convert the dataset into an optimized data structure 'Dmatrix' used by XGBoost
dm = xgb.DMatrix(data = X, label = y)

## Define cross-validation method: here time-series-specific strategy
tscv = TimeSeriesSplit(n_splits = 5)

## Set the number of processors/cores to be used
n_processors = os.cpu_count() - 1 # '-1' if all processors

#--------------------------------------------------------------------------------------------------
# STEP 1: Tuning 'n_estimators' and 'learning_rate'
#--------------------------------------------------------------------------------------------------

# Initialize the base model to tune
xgboost = xgb.XGBRegressor(objective = 'reg:linear',
                           max_depth = 5,
                           min_child_weight = 1,
                           gamma = 0,
                           colsample_bytree = 0.8,
                           subsample = 0.8
                          )

# Create the search space of hyperparameters
param_space = {'n_estimators': [int(x) for x in [100, 200, 500, 700, 1000]],#list(np.linspace(100, 1000, 3))],
               'learning_rate': [round(x, 2) for x in list(np.linspace(0.05, 0.3, 3))]
              }

# Initialize GRID search using n-fold cross-validation
grid_search = GridSearchCV(estimator = xgboost,
                           param_grid = param_space,
                           cv = tscv,
                           verbose = 2,
                           n_jobs = n_processors
                          )

# Run parameter search for optimal hyperparameters
start_time = time.time()
grid_search_outcome = grid_search.fit(X, y)
best_params = grid_search_outcome.best_params_

# Get the tuned hyper-paramaters of interest
tuned_n_estimators = best_params['n_estimators']
tuned_learning_rate = best_params['learning_rate']

# Summarize results of the grid search for optial hyperparameters
print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best: %f using %s" % (grid_search_outcome.best_score_, grid_search_outcome.best_params_))

#--------------------------------------------------------------------------------------------------
# STEP 2: Tuning 'max_depth' and 'min_child_weight'
#--------------------------------------------------------------------------------------------------

# Initialize the base model to tune
xgboost = xgb.XGBRegressor(objective = 'reg:linear',
                           max_depth = 5,
                           min_child_weight = 1,
                           gamma = 0,
                           colsample_bytree = 0.8,
                           subsample = 0.8,
                           n_estimators = tuned_n_estimators,
                           learning_rate = tuned_learning_rate
                          )
# Create the search space of hyperparameters
param_space = {'max_depth': [int(x) for x in [2, 3, 5, 8, 10, 12]],#list(np.linspace(3, 12, 5))],
               'min_child_weight': [int(x) for x in list(np.linspace(1, 7, 4))],
              }

# Initialize GRID search using n-fold cross-validation
grid_search = GridSearchCV(estimator = xgboost,
                           param_grid = param_space,
                           cv = tscv,
                           verbose = 2,
                           n_jobs = n_processors
                          )

# Run parameter search for optimal hyperparameters
start_time = time.time()
grid_search_outcome = grid_search.fit(X, y)
best_params = grid_search_outcome.best_params_

# Get the tuned hyper-paramaters of interest
tuned_max_depth = best_params['max_depth']
tuned_min_child_weight = best_params['min_child_weight']

# Summarize results of the grid search for optial hyperparameters
print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best: %f using %s" % (grid_search_outcome.best_score_, grid_search_outcome.best_params_))

#--------------------------------------------------------------------------------------------------
# STEP 3: Tuning 'gamma'
#--------------------------------------------------------------------------------------------------
# Initialize the base model to tune
xgboost = xgb.XGBRegressor(objective = 'reg:linear',
                           max_depth = tuned_max_depth,
                           min_child_weight = tuned_min_child_weight,
                           gamma = 0,
                           colsample_bytree = 0.8,
                           subsample = 0.8,
                           n_estimators = tuned_n_estimators,
                           learning_rate = tuned_learning_rate
                          )
# Create the search space of hyperparameters
param_space = {'gamma': [round(x, 1) for x in list(np.linspace(0.0, 1.0, 11))],
              }

# Initialize GRID search using n-fold cross-validation
grid_search = GridSearchCV(estimator = xgboost,
                           param_grid = param_space,
                           cv = tscv,
                           verbose = 2,
                           n_jobs = n_processors
                          )

# Run parameter search for optimal hyperparameters
start_time = time.time()
grid_search_outcome = grid_search.fit(X, y)
best_params = grid_search_outcome.best_params_

# Get the tuned hyper-paramaters of interest
tuned_gamma = best_params['gamma']

# Summarize results of the grid search for optial hyperparameters
print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best: %f using %s" % (grid_search_outcome.best_score_, grid_search_outcome.best_params_))

#--------------------------------------------------------------------------------------------------
# STEP 4: Tuning 'subsample' and 'colsample_bytree'
#--------------------------------------------------------------------------------------------------
# Initialize the base model to tune
xgboost = xgb.XGBRegressor(objective = 'reg:linear',
                           max_depth = tuned_max_depth,
                           min_child_weight = tuned_min_child_weight,
                           gamma = tuned_gamma,
                           colsample_bytree = 0.8,
                           subsample = 0.8,
                           n_estimators = tuned_n_estimators,
                           learning_rate = tuned_learning_rate
                          )
# Create the search space of hyperparameters
param_space = {'subsample': [round(x, 1) for x in list(np.linspace(0.5, 1.0, 6))],
               'colsample_bytree': [round(x, 1) for x in list(np.linspace(0.4, 1.0, 7))]
              }

# Initialize GRID search using n-fold cross-validation
grid_search = GridSearchCV(estimator = xgboost,
                           param_grid = param_space,
                           cv = tscv,
                           verbose = 2,
                           n_jobs = n_processors
                          )

# Run parameter search for optimal hyperparameters
start_time = time.time()
grid_search_outcome = grid_search.fit(X, y)
best_params = grid_search_outcome.best_params_

# Get the tuned hyper-paramaters of interest
tuned_subsample = best_params['subsample']
tuned_colsample_bytree = best_params['colsample_bytree']

# Summarize results of the grid search for optial hyperparameters
print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best: %f using %s" % (grid_search_outcome.best_score_, grid_search_outcome.best_params_))

#--------------------------------------------------------------------------------------------------
# STEP 5: Tuning regularization parameters (L1: 'alpha' or L2: 'lambda')
#--------------------------------------------------------------------------------------------------
# Initialize the base model to tune
xgboost = xgb.XGBRegressor(objective = 'reg:linear',
                           max_depth = tuned_max_depth,
                           min_child_weight = tuned_min_child_weight,
                           gamma = tuned_gamma,
                           colsample_bytree = tuned_colsample_bytree,
                           subsample = tuned_subsample,
                           n_estimators = tuned_n_estimators,
                           learning_rate = tuned_learning_rate
                          )
# Create the search space of hyperparameters
param_space = {'alpha': [1e-5, 1e-2, 0.1, 1, 100],
              }

# Initialize GRID search using n-fold cross-validation
grid_search = GridSearchCV(estimator = xgboost,
                           param_grid = param_space,
                           cv = tscv,
                           verbose = 2,
                           n_jobs = n_processors
                          )

# Run parameter search for optimal hyperparameters
start_time = time.time()
grid_search_outcome = grid_search.fit(X, y)
best_params = grid_search_outcome.best_params_

# Get the tuned hyper-paramaters of interest
tuned_alpha = best_params['alpha']

# Summarize results of the grid search for optial hyperparameters
print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best: %f using %s" % (grid_search_outcome.best_score_, grid_search_outcome.best_params_))
#--------------------------------------------------------------------------------------------------
# ToDo STEP 6 (FINAL MODEL): Generating more trees and reducing learning rate
#--------------------------------------------------------------------------------------------------

## Define custom base scoring function: Magnitude of relative errror
def magnitude_relative_error(y_true, y_predicted):
    if y_true == 0:
        return y_predicted
    else:
        return np.abs((y_predicted - y_true)/y_true)

## Define custom scoring function: Mean Magnitude of Relative Error (MMRE)
def mean_magnitude_relative_error(y_true, y_predicted):
    return np.mean(magnitude_relative_error(y_true, y_predicted))

## Define custom scoring function: Median Magnitude of Relative Error (MedMRE)
def median_magnitude_relative_error(y_true, y_predicted):
    return np.median(magnitude_relative_error(y_true, y_predicted))

## DEFINE SCORING FUNCTIONS for Evaluating Model's Performance
scoring = {'Explained_variance':    'explained_variance',
           'MAE':                   'neg_mean_absolute_error',
           'MSE':                   'neg_mean_squared_error',
           'MedAE':                 'neg_median_absolute_error'#,
#           'MMRE':                  make_scorer(mean_magnitude_relative_error, greater_is_better = False),
#           'MedMRE':                make_scorer(median_magnitude_relative_error, greater_is_better = False)
          }

# Initialize the base model to tune
xgboost = xgb.XGBRegressor(objective = 'reg:linear',
                           max_depth = tuned_max_depth,
                           min_child_weight = tuned_min_child_weight,
                           gamma = tuned_gamma,
                           colsample_bytree = tuned_colsample_bytree,
                           subsample = tuned_subsample,
                           n_estimators = tuned_n_estimators,
                           learning_rate = tuned_learning_rate,
                           alpha = tuned_alpha
                          )

## Define time-series cross-validation
tscv = TimeSeriesSplit(n_splits = 10)

## Train and test model in cross-validation
xgbr_cv = cross_validate(estimator = xgboost,
                         X = X, y = y, cv = tscv,
                         scoring = scoring,
                         return_train_score = True,
                         verbose = 2,
                         n_jobs = multiprocessing.cpu_count() - 1 # '-1' if all processors,
                        )

## Print model's performance
print("Trainig scores:")
print("Explained variance:", round(np.median(xgbr_cv['train_Explained_variance']), 5))
print("MAE:", round(np.median(xgbr_cv['train_MAE']), 5))
print("MSE:", round(np.median(xgbr_cv['train_MSE']), 5))
print("MedAE:", round(np.median(xgbr_cv['train_MedAE']), 5))
#print("MMRE:", round(np.median(xgbr_cv['train_MMRE']), 5))
#print("MedMRE:", round(np.median(xgbr_cv['train_MedMRE']), 5))



print("\nTesting scores:")
print("Explained variance:", round(np.median(xgbr_cv['test_Explained_variance']), 5))
print("MAE:", round(np.median(xgbr_cv['test_MAE']), 5))
print("MSE:", round(np.median(xgbr_cv['test_MSE']), 5))
print("MedAE:", round(np.median(xgbr_cv['test_MedAE']), 5))
#print("MMRE:", round(np.median(xgbr_cv['test_MMRE']), 5))
#print("MedMRE:", round(np.median(xgbr_cv['test_MedMRE']), 5))

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## FEATURE IMPROTANCE
## https://explained.ai/rf-importance/
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## Split data into train and test sets (Hold-out split: 70% Train, 30% Test)
hold_out = 0.3
split = len(X)-int(len(X) * hold_out)

X_train = X.iloc[0:split, ].copy()
y_train = y[0:split].copy()

X_test = X.iloc[split+1:, ].copy()
y_test = y[split+1:].copy()

## Fit random forest with best parameters
xgbr = xgboost.fit(X_train, y_train)

##-------------------------------------------------------------------------------------------------
## Get Default importances: based on ...
feature_importances = pd.DataFrame(xgbr.feature_importances_,
                                      index = X.columns,
                                      columns = ['importance']).sort_values('importance', ascending = False)

print ("Features sorted by their score:")
print (feature_importances)
sns.barplot(data = feature_importances, x = 'importance', y = feature_importances.index)

###################################################################################################
### EOF ###