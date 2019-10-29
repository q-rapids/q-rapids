##################################################################################################
# PURPOSE:  Integrate Git commit history data with Jira issue log data
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime.date, calendar, pandas (pd), numpy (np), seaborn (sns), re,
#           matplotlib.pyplot (plt),  matplotlib.ticker ticker), re, 
#           Please read and comply to the applicable usage licenses.
####################################################################################################

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## LOAD DATA
## /!\ 'Pickle' is a binary format so reading pickle data from untrusted sources can be unsafe
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

input_path = os.path.normpath("<location/directorry of input data>")
output_path = os.path.normpath("<location/directorry of output data>")
data_folder = "PICKLE"

## GIT-JIRA Data
git_jira_data_set = "<file name>"
git_jira_data_version = "file verions time stamp"
git_jira_data_variant = "file variant"

git_jira_data_source = os.path.join(input_path, 'PICKLE',
                                    git_jira_data_set +
                                    git_jira_data_version +
                                    git_jira_data_variant + '.pickle')
df_git_jira = pd.read_pickle(git_jira_data_source)

## SONAR-QUBE Data
sonar_data_set = "<>file name"
sonar_data_version = "<data version time stamp>"
sonar_data_variant = "<data variant>"

sonar_data_source = os.path.join(input_path, 'PICKLE',
                                    sonar_data_set +
                                    sonar_data_version +
                                    sonar_data_variant + '.pickle')
df_sonar = pd.read_pickle(sonar_data_source)

## Look into the data; 'dataframe_summary()' defined in own functions
basic_stats_git_jira = dataframe_summary(df_git_jira)
basic_stats_sonar = dataframe_summary(df_sonar)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## CLEAN DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## ------------------------------------------------------------------------------------------------
# ## REMOVE ITEMS for source code files of irrelevant programming languages (retain java only)
df_git_jira = df_git_jira[df_git_jira['File'].str.endswith('.java', na = False)]
df_sonar = df_sonar[df_sonar['File'].str.endswith('.java', na = False)]

## ------------------------------------------------------------------------------------------------
## Remove irrelevant features. Identify the minimal set of relevant features per data set

## Select through exclusion (currently all features selected)
git_jira_relevant_features = [x for x in df_git_jira.columns if x not in ['']]

## Select through exclusion (currently all features selected)
sonar_relevant_features = [x for x in df_sonar.columns if x not in ['']]

## Reduce data sets to the selected relevant features and rename features if appropriate
df_git_jira = df_git_jira[git_jira_relevant_features]
df_sonar = df_sonar[sonar_relevant_features]

##-------------------------------------------------------------------------------------------------
## Handle MISSING VALUES

## Set how much missing can maximally REMAIN per feature
max_missing = 0.2
missing_threshold = 1 - max_missing

## Identify features where %missing > 1-max_missing
print('Git-Jira features exceeding ', missing_threshold*100, '% of missing values:',
      df_git_jira.columns[
          (df_git_jira.isna().sum() / len(df_git_jira)) > (missing_threshold)].tolist())
print('SonarQube Metrics features exceeding ', missing_threshold*100, '% of missing values:',
      df_sonar.columns[
          (df_sonar.isna().sum() / len(df_sonar)) > (missing_threshold)].tolist())

## Remove features with majority of entries missing (if any such exists)
df_git_jira.dropna(thresh = len(df_git_jira) * max_missing, axis = 1, inplace = True)
df_sonar.dropna(thresh = len(df_sonar) * max_missing, axis = 1, inplace = True)

##-------------------------------------------------------------------------------------------------
## Handle NON-UNIQUE Features: Remove features with constant value

## View the list of features with constant values (no varaince)
print('Git-Jira features without variance: ',
      df_git_jira.columns[df_git_jira.apply(lambda col: col.unique().size == 1)].values)
print('SonarQube Metrics features without variance: ',
      df_sonar.columns[df_sonar.apply(lambda col: col.unique().size == 1)].values)

## Remove non-unique features, if any exist
df_git_jira = df_git_jira.drop(df_git_jira.columns[df_git_jira.
                               apply(lambda col: col.unique().size == 1)], axis = 1)
df_sonar = df_sonar.drop(df_sonar.columns[df_sonar.
                         apply(lambda col: col.unique().size == 1)], axis = 1)

##-------------------------------------------------------------------------------------------------
## Transform key features to consistent formatting

## Path, Component, Filename: Consistent format: src/.../<filename>.java
df_git_jira = df_git_jira.assign(File = df_git_jira['File'].apply(lambda x: x.split('/', 1)[1]))

##-------------------------------------------------------------------------------------------------
## Handle duplicates (here we DO NOT remove duplicates)

## Check for duplicates on specified features: here the key features to join data sets on
features_for_duplicates = ['Date', 'File']

duplicated_git_jira = df_git_jira.groupby(
    df_git_jira[features_for_duplicates].columns.tolist()).size().reset_index()\
    .rename(columns={0: 'duplicates'}).sort_values('duplicates', ascending=False)

print(duplicated_git_jira.head())

duplicated_sonar = df_sonar.groupby(
    df_sonar[features_for_duplicates].columns.tolist()).size().reset_index()\
    .rename(columns={0: 'duplicates'}).sort_values('duplicates', ascending=False)

print(duplicated_sonar.head())

## Sort data to investigate potential duplicates
## Duplicates are caused mainly be the fact that several different issues could be
## addressed in the same commit (time), in the same file.
df_sonar.sort_values(by = features_for_duplicates, ascending = False, inplace = True)
df_git_jira.sort_values(by = features_for_duplicates, ascending = False, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## MERGE Git-Jira with SonarQube
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## /!\ WARNING: Multiple duplicates will be created (for the same commit(date and file multiple
##              Sonar and multiple Jira issues are possible

## Join data sets
df_git_jira_sonar = pd.merge(left = df_sonar, right = df_git_jira, how = 'outer',
                    left_on = ['Date', 'File'], right_on = ['Date', 'File'],
                    suffixes=('_gitjira', '_sonar'))

df_git_jira_sonar.sort_values(by = features_for_duplicates, ascending = False, inplace = True)


## Check data
print("Git-Jira dimensions:", df_git_jira.shape)
print("SonarQube dimensions:", df_sonar.shape)
print("Integrate data dimensions:", df_git_jira_sonar.shape)

basisc_stats_gitjira_sonar = dataframe_summary(df_git_jira_sonar)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SAVE data to files
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

data_set = "<output data file name>"
data_version = "<data version time stamp>"
data_variant = "<data variant>"

## Save integrated data to a '.pickle' file
data_target = os.path.join(output_path, 'PICKLE',
                                    data_set +
                                    data_version +
                                    data_variant + '.pickle')
df_git_jira_sonar.to_pickle(data_target)

## Save preprocessed source data sets to files
## Set data storage path
git_jira_data_variant = "<data variant>"
git_jira_data_target = os.path.join(input_path, 'PICKLE',
                                    git_jira_data_set +
                                    git_jira_data_version +
                                    git_jira_data_variant + '.pickle')
df_git_jira.to_pickle(git_jira_data_target)

sonar_data_variant = "<data variant>"
sonar_data_target = os.path.join(input_path, 'PICKLE',
                                    sonar_data_set +
                                    sonar_data_version +
                                    sonar_data_variant + '.pickle')
df_sonar.to_pickle(sonar_data_target)


###################################################################################################
### EOF ###