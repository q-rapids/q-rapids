##################################################################################################
# PURPOSE:  Deriving Random Forest classification moodel from example data
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime (dt), time, sys, pickle, pandas (pd), numpy (np), scipy.stats,
#           collections.namedtuple, multiprocessing, itertools, matplotlib.pyplot (plt)
#           seaborn (sns),
#           imblearn.over_sampling.SMOTENC, imblearn.over_sampling.SMOTE, 
#           imblearn.over_sampling.RandomOverSampler, imblearn.under_sampling.RandomUnderSampler,
#           sklearn.model_selection.GridSearchCV, sklearn.model_selection.RandomizedSearchCV.
#           sklearn.model_selection.cross_validate, sklearn.model_selection. TimeSeriesSplit,
#           sklearn.model_selection.cross_val_score, sklearn.metrics.scorer.make_scorer,
#           sklearn.model_selection.cross_val_predict,
#           sklearn.metrics (mt), sklearn.ensemble.RandomForestClassifier
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
response_feature = 'Issue_Type_Bug'
predictor_features = [feat for feat in relevant_features if feat != response_feature]
predictor_features_cat = df[predictor_features].select_dtypes(include=[np.object]).columns.tolist()
predictor_features_num = df[predictor_features].select_dtypes(include=[np.number]).columns.tolist()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HANDLE MISSING DATA: replace missing data on numeric columns with '0'
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##ToDo-OPTIONAL: Drop rows where Issue_Type_Bug is NaN
df = df.dropna(subset = [response_feature]).copy()

num_features = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_features] = df[num_features].fillna(value = 0)#, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## DISCRETIZE RESPONSE VARIABLES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Visualize distribution of the response feature to discretize
print("Number of unique values of", response_feature, "feature:", df[response_feature].nunique())
print("List of unique values of", response_feature, "feature:", df[response_feature].unique())

sns.distplot(df[response_feature],
             bins = df[response_feature].nunique(),
             hist = True,
             kde = False,
             fit = stats.gamma,
             rug = True)

## DISCRETIZE response variable. Naive approach: each unique bug count ist coded as category
## Add discrete feature right after the numeric one
# response_feature_cat = response_feature_num + '_Cat'
# feature_index = df.columns.get_loc(response_feature_num)
# discretized_data = df[response_feature_num].astype(int).astype(str)
# df.insert(feature_index+1, response_feature_cat, discretized_data)

## (Alterantively) BINARIZE response variable. If #Bugs == 0 then '0' else '1'
## Add dichtomous feature right after the numeric one
mask = df[response_feature] > 0
df.loc[mask, response_feature] = 1

## For multi-class problems lebal should be converted to string (category)
## For binary-class problems labels should be integer: 1-positive class, 0-negative class
#df[response_feature] = df[response_feature].astype(int).astype(str)

## Visualize histogram of the response feature after discretization
plt = count_plot(data = df,
                 feature = response_feature,
                 title = "Distribution of Bugs Found (per Code File, over Time, Daily)",
                 label = "Number of issues of type 'Bug'",
                 annotation = 'count')


## Fit discrete Poisson and Negative-Binomial distributions
# data = df[response_feature_cat].astype(int).copy()
# k = np.arange(data.max()+1)
# mlest = data.mean()
# plt.plot(k, stats.poisson.pmf(k, mlest)*len(data), 'go', markersize = 8)
# plt.plot(k, stats.nbinom.pmf(k, mlest)*len(data), 'go', markersize = 8)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## BALANCE Data
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Split data for balacing
y = df[response_feature].copy()
X = df[predictor_features].copy()

## Remember column names and types (because balancing method will loose them)
cols_dict = {}
for column in X.columns:
        cols_dict.update({column:X[column].dtype})

## Set the number of cases to be oversampled
balance_count = np.count_nonzero(y == 0)

## Oversample minority class up to number of majority-class observations
categorical_features = [X.columns.get_loc(c) for c in predictor_features_cat if c in X]

sampling_strategy = {1: balance_count, 0:balance_count}
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
## DEFINE SCORING FUNCTIONS for Evaluating Model's Performance
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

scoring = {'Accuracy':              'accuracy',             # weak measure for imbalanced data
           'Precision':             'precision',            # binary-class
           'Recall':                'recall',               # binary-class
           'F1-Score':              'f1',                   # binary-class
          }

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SEARCH FOR OPTIMAL HYPERPARAMETERS
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## GRID SEARCH optimization: Search space based on the outcomes of random search

##Create the search space of hyperparameters: based on best parameters found in random search
param_space = {'n_estimators': [200, 500, 1000],
               'max_features': ['sqrt'],
               'min_samples_split': [10, 20, 50, 70, 100],
               'min_samples_leaf': [10, 20, 50, 100, 300, 500],
               'bootstrap': [True],
               'oob_score': [True]
              }

# Initialize the base model to tune
random_forest = RandomForestClassifier()

## Define cross-validation method: here time-series-specific strategy
tscv = TimeSeriesSplit(n_splits = 5)

# Initialize GRID search using n-fold cross-validation
grid_search = GridSearchCV(estimator = random_forest,
                           param_grid = param_space,
                           cv = tscv,
                           n_jobs = multiprocessing.cpu_count() - 1, # '-1' if all processors,
                           verbose = 2
                          )

# Run parameter search for optimal hyperparameters
start_time = time.time()
grid_result = grid_search.fit(X, y)

# Summarize results of the grid search for optial hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## RUN RANDOM FOREST IN CROSS-VALIDATION
## https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Select best parameters for learning final model
# Q-Rapids DD: 'bootstrap': True, 'max_features': 'sqrt', 'min_samples_leaf': 100, 'min_samples_split': 20, 'n_estimators': 200, 'oob_score': True
best_params = grid_result.best_params_

# Initialize random forest model with best parameters found in tuning
random_forest = RandomForestClassifier(n_estimators = best_params['n_estimators'],
                                       max_features = best_params['max_features'],
                                       min_samples_leaf = best_params['min_samples_leaf'],
                                       min_samples_split = best_params['min_samples_split'],
                                       bootstrap = best_params['bootstrap'],
                                       oob_score = best_params['oob_score']
                                      )

## Define time-series cross-validation
tscv = TimeSeriesSplit(n_splits = 5)

## Define custom scoring function: cust_scoreing = make_scorer(
## score_func = cusome-defined scoring function score_func(y, y_pred, **kwargs)
## greater_is_better =  whther large values of score_func are good or bad
## Further params: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

## Define scoring functions using parformance metrics predefined in sklearn
## https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
## Notice: 'neg_' functino return negative values (interpreted as loss).
## Use 'abs' function if positive values are reuired.
# print('List of sklearn scoring functions (by key name):')
# for score in sorted(mt.SCORERS.keys()):
#     print ('\t', score)

## Train and test model in cross-validation
rfc_cv = cross_validate(estimator = random_forest,
                        X = X, y = y, cv = tscv,
                        scoring = scoring,
                        return_train_score = True,
                        verbose = 2,
                        n_jobs = multiprocessing.cpu_count() - 1 # '-1' if all processors,
                       )

## Print model's performance
print("\nTesting scores for multi-class problem:")
print("Accuracy:", round(np.median(rfc_cv['test_Accuracy']), 3))
print("Precision:", round(np.median(rfc_cv['test_Precision']), 3))
print("Recall:", round(np.median(rfc_cv['test_Recall']), 3))
print("F1-Score:", round(np.median(rfc_cv['test_F1-Score']), 3))

## Confusion Matrix
y_pred = cross_val_predict(random_forest, X = X, y = y, cv = 10)
ax = sns.countplot(y_pred)

np.set_printoptions(precision=2)
class_names = df[response_feature].unique()
# Plot non-normalized confusion matrix
plot_confusion_matrix(y, y_pred, classes = class_names,
                      title='Confusion matrix, without normalization')

print(mt.classification_report(y, y_pred))

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## LEARN FINAL MODEL ON COMPLETE DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## Learn model on the complete dataset
rfc = RandomForestClassifier(n_estimators = best_params['n_estimators'],
                                       max_features = best_params['max_features'],
                                       min_samples_leaf = best_params['min_samples_leaf'],
                                       min_samples_split = best_params['min_samples_split'],
                                       bootstrap = best_params['bootstrap'],
                                       oob_score = best_params['oob_score']
                                      )
rfc.fit (X = X, y = y)

##-------------------------------------------------------------------------------------------------
## Get Impurity-based importances: based on the nodes' local impurities (Sklearn's standard method)
# /!\ Impurity-based importances are biased, e.g., once one of correlated predictors is selected in a
#     node the others get low importance
feature_importances = pd.DataFrame(rfc.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

print ("Features sorted by their score:")
print (feature_importances)
sns.barplot(data = feature_importances, x = 'importance', y = feature_importances.index)

##-------------------------------------------------------------------------------------------------
## Select the most important features and reduce model down to those features only
importance_threshold = 0.03
important_features = feature_importances.loc[feature_importances['importance'] >= importance_threshold].index
#important_features = []
X = X[important_features].copy()

## Confusion Matrix
y_pred = cross_val_predict(random_forest, X = X, y = y, cv = 10)
ax = sns.countplot(y_pred)

np.set_printoptions(precision=2)
class_names = df[response_feature].unique()
# Plot non-normalized confusion matrix
plot_confusion_matrix(y, y_pred, classes = class_names,
                      title='Confusion matrix, without normalization')

## Learn model on compelte data for the most important features only
rfc.fit (X = X, y = y)

##-------------------------------------------------------------------------------------------------
## Save model for deployment
python_version = sys.version.split(" ", 1)[0]

data_target = os.path.join(output_path, 'Models',
                           data_set +
                           data_version +
                           data_variant +
                           '_CompleteDataBalancedBinaryRandomForestClassModel_Python' +
                           python_version +
                           '.pickle')
pickle.dump(rfc, open(data_target, 'wb'))

## Test whether after loading the model will behave as expected
model = pickle.load(open(data_target, 'rb'))

rfc.predict(X[0:1,])
model.predict(X[0:1,])

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## FEATURE IMPORTANCE
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
rfc = random_forest.fit(X_train, y_train)

##-------------------------------------------------------------------------------------------------
## Get Impurity-based importances: based on the nodes' local impurities (Sklearn's standard method)
# /!\ Impurity-based importances are biased, e.g., once one of correlated predictors is selected in a
#     node the others get low importance
feature_importances = pd.DataFrame(rfc.feature_importances_,
                                      index = X.columns,
                                      columns = ['importance']).sort_values('importance', ascending = False)

print ("Features sorted by their score:")
print (feature_importances)
sns.barplot(data = feature_importances, x = 'importance', y = feature_importances.index)

