##################################################################################################
# PURPOSE:  Integratiopn of SonarQube data sets
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime.date, calendar, pandas (pd), numpy (np), seaborn (sns), 
#           matplotlib.pyplot (plt),  matplotlib.ticker ticker), re, 
#           Please read and comply to the applicable usage licenses.
####################################################################################################

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## LOAD DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Set data storage path
input_path = os.path.normpath("<location/directory of input data>")
output_path = os.path.normpath("location/directory of output data ")
data_folder = "PICKLE"

sonar_metrics_data_set = "<data file name>"
sonar_metrics_data_version = "<data version time stamp>"
sonar_metrics_data_variant = "<data variant>"

sonar_metrics_data_source = os.path.join(input_path, 'PICKLE',
                                 sonar_metrics_data_set +
                                 sonar_metrics_data_version +
                                 sonar_metrics_data_variant + '.pickle')
df_sonar_metrics = pd.read_pickle(sonar_metrics_data_source)


sonar_issues_data_set = "<data file name>"
sonar_issues_data_version = "<data version time stamp>"
sonar_issues_data_variant = "<data variant>"

sonar_issues_data_source = os.path.join(input_path, 'PICKLE',
                                 sonar_issues_data_set +
                                 sonar_issues_data_version +
                                 sonar_issues_data_variant + '.pickle')
df_sonar_issues = pd.read_pickle(sonar_issues_data_source)


##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## CLEAN DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## Handle MISSING VALUES

## Set how much missing can maximally REMAIN per feature
max_missing = 0.2
missing_threshold = 1 - max_missing

## Identify features where %missing > missing_threshold
print('SonarQube Metrics features exceeding', missing_threshold*100, '% of missing values:',
      df_sonar_metrics.columns[
          (df_sonar_metrics.isna().sum() / len(df_sonar_metrics)) > (missing_threshold)].tolist())
print('SonarQube Issues features exceeding', missing_threshold*100, '% of missing values:',
      df_sonar_issues.columns[
          (df_sonar_issues.isna().sum() / len(df_sonar_issues)) > (missing_threshold)].tolist())

## Remove features where %missing > missing_threshold (if any such exists)
df_sonar_metrics.dropna(thresh = len(df_sonar_metrics) * max_missing, axis = 1, inplace = True)
df_sonar_issues.dropna(thresh = len(df_sonar_issues) * max_missing, axis = 1, inplace = True)

##-------------------------------------------------------------------------------------------------
## Handle NON-UNIQUE Features: Remove features with constant value

## View the list of features with constant values (no variance)
print('SonarQube Metrics features without variance: ',
      df_sonar_metrics.columns[df_sonar_metrics.apply(lambda col: col.unique().size == 1)].values)
print('SonarQube Issues features without variance: ',
      df_sonar_issues.columns[df_sonar_issues.apply(lambda col: col.unique().size == 1)].values)

## Remove non-unique features
df_sonar_metrics = df_sonar_metrics.drop(df_sonar_metrics.columns[df_sonar_metrics.
                                         apply(lambda col: col.unique().size == 1)], axis = 1)
df_sonar_issues = df_sonar_issues.drop(df_sonar_issues.columns[df_sonar_issues.
                                       apply(lambda col: col.unique().size == 1)], axis = 1)

##-------------------------------------------------------------------------------------------------
## Transform key features to consistent formatting

## Path, Component, Filename: Consistent format: src/.../<filename>.java
df_sonar_issues = df_sonar_issues.assign(
    File = df_sonar_issues['File'].apply(lambda x: (re.sub('.*:', '', x))))

## ------------------------------------------------------------------------------------------------
## Remove irrelevant features and aggregate data per unique groups of items (for key-features)

## Select through exclusion
sonar_metrics_relevant_features = [x for x in df_sonar_metrics.columns if x not in []]

## Select through exclusion
## The 'Key', Line' and 'Offset' features are not interesting for quality analysis. yet, they create
## a secondary key because for the same commit, date, and file multiple issues of the same type
## can and are reported but in different lines of code-
## Discarding these features leads to duplicated data items,w chih must be handled afterwards
sonar_issues_relevant_features = [x for x in df_sonar_issues.columns if x not in [
                                  #'key',
                                  'Start_Line',
                                  'End_Line',
                                  'Start_Offset',
                                  'End_Offset',
                                  'Line',
                                  'Message',
                                  'Debt'
                                 ]]

## Reduce data sets to the selected relevant features
df_sonar_metrics = df_sonar_metrics[sonar_metrics_relevant_features]
df_sonar_issues = df_sonar_issues[sonar_issues_relevant_features]

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## AGGREGATE DATA Duplicates
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Problem: Multiple issues of the same type are reported for the same commit, date and file
## (the same issue occurs in diffrent lines of code and all are handled in the same commit)
## Solution: Informative features will be aggregated on key-features
## In this case  data is groupped on all features but 'linesCount' (#lines affected by an issue)
## and 'liensCount' is summed up for each unique group
aggregation_functions = {'Lines_Count':     'sum',
                         'Severity_Count':  'sum',
                         'Effort':          'sum',
                         'Key':             'count'
                        }
groupping_features = [x for x in df_sonar_issues.columns if x not in list(aggregation_functions.keys())]

df_sonar_issues = df_sonar_issues.groupby(groupping_features, as_index = False).agg(aggregation_functions)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## RENAME FEATURES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Name key features consistently so that outer merge created proper union
## (no missing values on key features)
df_sonar_issues.rename(columns = {'Key':'Issue_Count'}, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HANDLE DUPLICATES (remaining)
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Specify features onwhich duplicates shall be identified
features_for_duplicates = ['Date', 'Commit_ID', 'File']

## Count duplicates on features to be used as key for integration
duplicated_sonar_metrics = df_sonar_metrics.groupby(
    df_sonar_metrics[features_for_duplicates].columns.tolist()).size().reset_index().rename(columns={0: 'duplicates'}) \
    .sort_values('duplicates', ascending=False)
print(duplicated_sonar_metrics.head())

duplicated_sonar_issues = df_sonar_issues.groupby(
    df_sonar_issues[features_for_duplicates].columns.tolist()).size().reset_index().rename(columns={0:'duplicates'})\
    .sort_values('duplicates', ascending = False)
print(duplicated_sonar_issues.head())

## Duplicates exist in that the same issue is reported with different creation data and status
## Remove duplicates:
## STEP 1: Select most recently created issue

if duplicated_sonar_issues.loc[0, 'duplicates'] > 1:
    ## Step 1: Select most recently created issues only
    df_sonar_issues.sort_values(by = ['Date', 'Commit_ID', 'File', 'Rule', 'Creation_Date'],
                                ascending = False, inplace = True)
    df_sonar_issues.drop_duplicates(subset = ['Date', 'Commit_ID', 'File', 'Rule'],
                                    keep='first', inplace=True)
    ## Step 2:

if duplicated_sonar_metrics.loc[0, 'duplicates'] > 1:
    df_sonar_metrics.sort_values(by = ['Date', 'Commit_ID', 'File'], ascending = False, inplace = True)
    df_sonar_metrics.drop_duplicates(keep = 'first', inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## UNFOLD ISSUE DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## (1) Unfold per Rule ID: New feature is created for each Rule and number of issues for given rule
##     is counted per each file and data/commit

index_features = ['Date', 'Commit_ID', 'File']
pivot_features = 'Rule'
value_features = 'Issue_Count'

df_issues_rule = df_sonar_issues.pivot_table(values = value_features,
                                             index = index_features,
                                             columns = pivot_features,
                                             aggfunc = np.sum,
                                             fill_value = 0
                                            )
df_issues_rule.reset_index(inplace = True)
df_issues_rule.columns = df_issues_rule.columns.str.replace('[^a-zA-Z0-9]', '_')

##-------------------------------------------------------------------------------------------------
## (2) Unfold per Rule Severity: New feature is created per eacu severity level and the number of
##     issues for given severity level is counted for each file and date/commit

index_features = ['Date', 'Commit_ID', 'File']
pivot_features = 'Severity'
value_features = 'Issue_Count'

df_issues_severity = df_sonar_issues.pivot_table(values = value_features,
                                                 index = index_features,
                                                 columns = pivot_features,
                                                 aggfunc = np.sum,
                                                 fill_value = 0
                                                )

df_issues_severity.reset_index(inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## MERGE DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## Join metrics with severity-pivoted data
df_sonar_sev = pd.merge(left = df_sonar_metrics, right = df_issues_severity, how = 'left',
                        left_on = ['Date', 'Commit_ID', 'File'],
                        right_on = ['Date', 'Commit_ID', 'File'],
                        suffixes=('_metrics', '_issues'))

print("Size of metrics data:", df_sonar_metrics.shape)
print("Size of issues data:", df_issues_severity.shape)
print("Size of merged data:", df_sonar_sev.shape)

##-------------------------------------------------------------------------------------------------
## Join metrics with rule-pivoted data
df_sonar_rul = pd.merge(left = df_sonar_metrics, right = df_issues_rule, how = 'left',
                        left_on = ['Date', 'Commit_ID', 'File'],
                        right_on = ['Date', 'Commit_ID', 'File'],
                        suffixes=('_metrics', '_issues'))

print("Size of metrics data:", df_sonar_metrics.shape)
print("Size of issues data:", df_issues_rule.shape)
print("Size of merged data:", df_sonar_rul.shape)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## VALIDATE MERGED DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Get summary stats (custom-defined function in 'Functions.py')
summary_stats_sev = dataframe_summary(df_sonar_sev)
summary_stats_rul = dataframe_summary(df_sonar_rul)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SAVE data to files
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Save integrated data to a '.pickle' file
sonar_merged_sev_path = os.path.join(output_path, 'PICKLE',
                                     '<output file name>.pickle')
df_sonar_sev.to_pickle(sonar_merged_sev_path)

sonar_merged_rul_path = os.path.join(output_path, 'PICKLE',
                                     '<output file name>.pickle')
df_sonar_rul.to_pickle(sonar_merged_rul_path)


## Save cleaned Sonar-Issues data
issues_clean_path = os.path.join(output_path, 'PICKLE',
                           '<output file name>.pickle')
df_sonar_issues.to_pickle(issues_clean_path)

## Save pivoted Sonar-Issues data
issues_pivot_sev_path = os.path.join(output_path, 'PICKLE',
                           '<output file name>.pickle')
df_issues_severity.to_pickle(issues_pivot_sev_path)

issues_pivot_rule_path = os.path.join(output_path, 'PICKLE',
                           '<output file name>.pickle')
df_issues_rule.to_pickle(issues_pivot_rule_path)


###################################################################################################
### EOF ###