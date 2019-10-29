##################################################################################################
# PURPOSE:  Preparation of the integrated data set for the analysis 
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime.date, calendar, pandas (pd), numpy (np), collections.namedtuple,
#           scipy.stats.stats (sc), statistics.mode 
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
gitjira_data_set = "<file name>"
gitjira_data_version = "<data version time stamp>"
gitjira_data_variant = "<data variant>"

gitjira_data_source = os.path.join(input_path, 'PICKLE',
                                   gitjira_data_set +
                                   gitjira_data_version +
                                   gitjira_data_variant + '.pickle')
df_gitjira = pd.read_pickle(gitjira_data_source)
#stats_gitjira = dataframe_summary(df_gitjira)

## SONAR-QUBE METRICS & ISSUES Data
sonar_data_set = "<file name>"
sonar_data_version = "<data version time stamp>"
sonar_data_variant = "<data variant>"

sonar_metrics_data_source = os.path.join(input_path, 'PICKLE',
                                 sonar_data_set +
                                 sonar_data_version +
                                 sonar_data_variant + '.pickle')
df_sonar = pd.read_pickle(sonar_metrics_data_source)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## CLEAN-UP DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## Remove none-'java' files
df_gitjira = df_gitjira[df_gitjira['Commit_File'].str.endswith('.java', na = False)].copy()
df_sonar = df_sonar[df_sonar['File'].str.endswith('.java', na = False)].copy()

## Truncate file path on the 'iese' directory; path begins with 'dd/...'
df_gitjira = df_gitjira.assign(Commit_File = df_gitjira['Commit_File'].apply(lambda x: x.split('iese', 1)[1]))
df_sonar = df_sonar.assign(File = df_sonar['File'].apply(lambda x: x.split('iese', 1)[1]))

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## Handle MISSING DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Missing data in Git-Jira data set
## Replace missing data on categorical features (e.g, Issue-Type, Issue-Priority) by 'Unknown'
missing_features = ['Issue_Key', 'Issue_Type', 'Issue_Priority', 'Issue_Component', 'Issue_Status',
                    'Issue_Platform', 'Issue_Reporter', 'Issue_Assignee']
df_gitjira[missing_features] = df_gitjira[missing_features].fillna('Unknown')

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## Handle NON-UNIQUE Features: Remove features with constant value
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## View the list of features with constant values (no variance)
print('GitJira features without variance: ',
      df_gitjira.columns[df_gitjira.apply(lambda col: col.unique().size == 1)].values)
print('SonarQube features without variance: ',
      df_sonar.columns[df_sonar.apply(lambda col: col.unique().size == 1)].values)

## Remove non-unique features
df_gitjira.drop(df_gitjira.columns[df_gitjira.
                apply(lambda col: col.unique().size == 1)], axis = 1, inplace = True)
df_sonar.drop(df_sonar.columns[df_sonar.
              apply(lambda col: col.unique().size == 1)], axis = 1, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SELECT RELEVANT FEATURES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Select features by excluding irrelevant ones

## Since we are interested on the dependency between code quality at certain time and the number
## and type of issue found (created) at that time, the Issue-Creation-Date is the only relevant
## date for this analysis purpose (to integrate GitJira data with SonarMetrics)
df_gitjira.insert(loc = 0, column = 'Date', value = df_gitjira['Issue_Creation_Date'])

## Remove entries on which 'Date' is missing
df_gitjira.dropna(subset = ['Date'], inplace = True)

## Select relevant (remove irrelevant) Git-Jira features
gitjira_relevant_features = [x for x in df_gitjira.columns if x not in [
    'Commit_Date',
    'Issue_Resolution_Date',
    'Issue_Update_Date',
    'Issue_Creation_Date'
]]
df_gitjira = df_gitjira[gitjira_relevant_features].copy()

## Rename Git-Jira columns if neccesary / for convenience
df_gitjira.rename(columns = {'Commit_File':'File'}, inplace=True)

## Select relevant (remove irrelevant) SonarQube features
sonar_relevant_features = [x for x in df_sonar.columns if x not in []]
df_sonar = df_sonar[sonar_relevant_features].copy()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## ENGINEER NEW FEATURES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Extract 'Date' from 'Date-Time' feature (rename source feature Date->DateTime)
## Git-Jira Data
df_gitjira.rename(columns={'Date':     'DateTime'},
                  inplace=True)

dt_col_index = df_gitjira.columns.get_loc("DateTime")
dt_data = pd.DatetimeIndex(df_gitjira['DateTime'])
df_gitjira.insert(dt_col_index+1, 'Date', dt_data.date)
df_gitjira.insert(dt_col_index+2, 'Time', dt_data.time)
df_gitjira.sort_values(['Date', 'File'], ascending = True, inplace=True)
print ("Number of unique dates in Git-Jira data:", df_gitjira['Date'].nunique())

## SonarQube Metrics Data
df_sonar.rename(columns = {'Date':     'DateTime'},
                inplace = True)

dt_col_index = df_sonar.columns.get_loc("DateTime")
dt_data = pd.DatetimeIndex(df_sonar['DateTime'])
df_sonar.insert(dt_col_index+1, 'Date', dt_data.date)
df_sonar.insert(dt_col_index+2, 'Time', dt_data.time)
df_sonar.sort_values(['Date', 'File'], ascending = True, inplace=True)
print ("Number of unique dates in SonrQube Metrics data:", df_sonar['Date'].nunique())

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SELECT RELEVANT FEATURES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Remove DateTime and Time features
irrelevant_features = ['DateTime']
df_gitjira.drop(irrelevant_features, axis = 1, inplace = True)
df_sonar.drop(irrelevant_features, axis = 1, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SELECT RELEVANT OBJECTS
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Select common time period for Git-Jira and SonarQube data
## Example: SonarQube data collected since 2017.07. / Git_Jira collected since 2015.08.

## Get the earliest and the latest common dates for the data sets considered
common_dates = list(set(
    df_gitjira['Date'].unique()).intersection(df_sonar['Date'].unique()))
start_date = min(common_dates)
end_date = max(common_dates)
print ("Common time period for GitJira and SonarQube is from", start_date, "to", end_date)

## Select data for common period of time, i.e., xclude data before the earliest and
## after latest common date
df_gitjira = df_gitjira[(df_gitjira['Date'] >= start_date) & (df_gitjira['Date'] <= end_date)].copy()
df_sonar = df_sonar[(df_sonar['Date'] >= start_date) & (df_sonar['Date'] <= end_date)].copy()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HANDLE OUTLIERS
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## Exclude data points, which correspond to renaming or moving large parts of SW project
## (a huge change in terms of added & deleted files and LOC)

## Identify time stamps (dates) for the Git-Jira outliers (extremely large chages of LOC)
outlier_features = ['Commit_Added_LOC', 'Commit_Deleted_LOC']
outlier_dates = df_gitjira.loc[~(np.abs(sc.zscore(df_gitjira[outlier_features])) < 3).any(axis=1), 'Date'].unique()

## Exclude outliers from the Git-Jira and SonarQube data sets
df_gitjira = df_gitjira.loc[~df_gitjira['Date'].isin(outlier_dates)].copy()
df_sonar = df_sonar.loc[~df_sonar['Date'].isin(outlier_dates)].copy()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## TRANSFORM GIT-JIRA DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Select relevant features
gitjira_relevant_features = [
    'Date',
    'File',
    'Commit_Added_LOC',
    'Commit_Deleted_LOC',
    'Commit_Issue_Count',
    'Issue_Type',
    'Issue_Priority'
]
df_gitjira = df_gitjira[gitjira_relevant_features].copy()

##-------------------------------------------------------------------------------------------------
## OPTION 1: Aggregate GitJira data on 'Date' and 'File'

## Encode categorical features
categorical_features = ['Issue_Type', 'Issue_Priority']
one_hot = pd.get_dummies(data = df_gitjira[categorical_features],
                         prefix = categorical_features, dtype = int)
df_gitjira.drop(categorical_features, axis = 1, inplace = True)
df_gitjira = df_gitjira.join(one_hot)

## Aggregate data; Sum up all values (dummies will now mean count of issue per type/priority
groupping_features = ['Date', 'File']

#groupping_features = [x for x in df_git.columns if x not in list(aggregation_functions.keys())]
df_gj_aggr = df_gitjira.groupby(groupping_features, as_index = False).agg('sum')
#df_gj_aggr.columns = df_gj_aggr.columns.droplevel(1)
#df_gj_aggr.columns = df_gj_aggr.columns.map('_'.join)

##-------------------------------------------------------------------------------------------------
## OPTION 2: Unfold 'Issue Type' to be able to integrate data per: day, file and issue type
## 'pivot_table' aggregates the data on the non-unique values of attributes listed for 'index'
## parameter. --> separate aggregation is not necessary anymore (data is unfolded on 'Issue Type'
## and aggregated on 'Date' and 'File'
# df_gj_unfold = df_gitjira.pivot_table(
#     index = ['Date', 'File'],
#     columns = ['Issue_Type'],
#     values = ['Commit_Added_LOC', 'Commit_Deleted_LOC', 'Commit_Issue_Count'],
#     aggfunc='sum').fillna(0)

## Convert index (joined Date and File) back into data frame columns
#df_gj_unfold.reset_index(level=['Date', 'File'], inplace=True)

## Flatten multilevel column index created by 'pivot_table' function
#df_gj_unfold.columns = [' '.join(col).strip() for col in df_gj_unfold.columns.values]

## Remove space in the column name introduced by flattening multilevel index
#df_gj_unfold.rename(columns = {'Date ': 'Date', 'File ': 'File'}, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## TRANSFORM SONAR-QUBE DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Generalize data to daily granularity: Select the most recent value per day
features_for_duplicates = ['File', 'Date']
duplicated_sonar = df_sonar.groupby(
    df_sonar[features_for_duplicates].columns.tolist()).size().reset_index()\
    .rename(columns={0: 'duplicates'}).sort_values('duplicates', ascending=False)
print(duplicated_sonar.head())

if duplicated_sonar.loc[0, 'duplicates'] > 1:
    df_sonar.sort_values(by = ['File', 'Date', 'Time'], ascending = False, inplace = True)
    df_sonar.drop_duplicates(subset = features_for_duplicates, keep = 'first', inplace = True)

## Drop irrelevant features
irrelevant_features = ['Time']
df_sonar.drop(irrelevant_features, axis = 1, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## INTEGRATE GitJira with SonarMetrics: Daily measurement and issue data per file
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## Left Join: SonarMetrics <-left- GitJira
## Every day has measurement data but only some days have issues)
# df_merged = pd.merge(left = df_sonar_metrics, right = df_gj_aggr, how = 'left',
#                     left_on = ['Date', 'File'], right_on = ['Date', 'File'],
#                     suffixes=('_issues', '_metrics'))


## Check how many dates from GitJira can be mapped to dates in SonarMetrics
common_dates = list(set(df_gj_aggr['Date'].unique()).intersection(df_sonar['Date'].unique()))
print("Unique dates in SonarMetrics:", len(df_sonar['Date'].unique()))
print("Unique dates in GitJira:", len(df_gj_aggr['Date'].unique()))
print("Unique date in common:", len(common_dates))

##-------------------------------------------------------------------------------------------------
## STEP 1: Map issues on their creation date to the nearest Sonar entry no later than
## the the issue creation date OR to any nearest Sonar entry within specific max time range?

## PREREQUISITE: Sort both data sets on key features: Date, File
df_sonar.sort_values(by = ['Date', 'File'], ascending = True, inplace = True)
df_sonar.reset_index(drop = True, inplace = True)
df_gj_aggr.sort_values(by = ['Date', 'File'], ascending = True, inplace = True)
df_gj_aggr.reset_index(drop = True, inplace = True)

## Add Mapping ID column to GitJira data set. It will be filled with index-IDs of the Sonar
## entries that are equal on 'File' and closest on 'Date'
df_gj_aggr['Mapping_ID'] = np.NaN

## Setr threshold for maximal time difference between Measurement entry and the following issue
time_threshold = pd.Timedelta('10 day')

## Get total number of loops for progras tracking purpose
issues_count = df_gj_aggr.shape[0]

## Run mapping
for i, row in df_gj_aggr.iterrows():
    print("Processing issue", i+1, "out of", issues_count)
    analogs = df_sonar[df_sonar['File'] == df_gj_aggr.loc[i, 'File']]
    if not analogs.empty:
        ## deltas will retain original index-IDs of SonarQube entries
        deltas = analogs['Date'] - df_gj_aggr.loc[i, 'Date']
        deltas = deltas[deltas <= dt.timedelta(days=0)]
        if (len(deltas) > 0 and abs(deltas.iloc[-1]) <= time_threshold):
            df_gj_aggr.at[i, 'Mapping_ID'] = deltas.index[-1]
## END of for

## Print out mapping results
mapped_all = df_gj_aggr['Mapping_ID'].count()
mapped_unique = df_gj_aggr['Mapping_ID'].nunique()
not_mapped = issues_count - mapped_all

print("Mapping results:")
print("\tTotal:\t\t", issues_count)
print("\tMapped:\t\t", mapped_all)
print("\tUniquely:\t", mapped_unique)
print( "\tNot mapped:\t", not_mapped)

##-------------------------------------------------------------------------------------------------
## STEP 2: Remove issues that cannot be mapped to any measurement

df_gj_aggr.dropna(subset = ['Mapping_ID'], inplace = True)
df_gj_aggr['Mapping_ID'] = df_gj_aggr['Mapping_ID'].astype(int)

##-------------------------------------------------------------------------------------------------
## STEP 3: Aggregate issues that map to the same measurment entry
## If an issue maps to more than one measurement entry (i.e., any duplicates on Mapping_ID)

## Select relevant features: issue impact and counts
## Exclude date and file name as irrelevant for aggregatino with sonar measurements
df_gj_aggr.drop(labels = ['Date', 'File'], axis = 1, inplace = True)

## IF more than one issue maps to the same measurement THEN aggregate them and set mapping to
## measurement indices as issues index ELSE only set the issue index to mapped measurement indices
if (df_gj_aggr['Mapping_ID'].nunique() < df_gj_aggr['Mapping_ID'].count()):
    df_gj_aggr = df_gj_aggr.groupby(['Mapping_ID'], sort = False, as_index = True).agg('sum')
else:
    df_gj_aggr.set_index(keys = ['Mapping_ID'], drop = True, inplace = True, verify_integrity = True)

##-------------------------------------------------------------------------------------------------
## STEP 4: Merge measurement (SonarQube) and issue (GitJira) data (use index as key)
df_merged = pd.merge(left = df_sonar, right = df_gj_aggr,
                     how = 'left', left_index = True, right_index = True,
                     suffixes = ('_sonar', '_gitjira'))

## Check dimensions of data sets before and after integration
print('Size of SonarQube:', df_sonar.shape)
print('Size of GitJira:', df_gj_aggr.shape)
print('Size of merged GitJira-SonarMetrics data:', df_merged.shape)

## Look into basic characteristics of the data frame
data_summary = dataframe_summary(df_merged)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## ENGINEER FEATURES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

insert_index = 2
df_merged.insert(insert_index+1, 'WeekDay', df_merged['Date'].apply(lambda x: calendar.day_name[x.weekday()]))
df_merged.insert(insert_index+2, 'Month', df_merged['Date'].apply(lambda x: calendar.month_name[x.month]))


##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SORT DATA on Date and FileName
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

df_merged.sort_values(by = ['Date', 'File'], ascending = True, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SAVE DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

data_set = "<file name>"
data_version = "<data version time stamp>"
data_variant = "<data variant>"

## Save integrated data to a '.pickle' file
data_target = os.path.join(output_path, 'PICKLE',
                           data_set +
                           data_version +
                           data_variant + '.pickle')
df_merged.to_pickle(data_target)


###################################################################################################
### EOF ###