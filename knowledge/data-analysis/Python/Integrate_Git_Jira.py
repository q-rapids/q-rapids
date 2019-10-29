##################################################################################################
# PURPOSE:  Integrate Git commit history data with Jira issue log data
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime.date, calendar, pandas (pd), numpy (np), seaborn (sns), statistics.mode
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

## Set data storage path
git_data_source = os.path.join(input_path, data_folder,
                               '<file name>.pickle')
jira_data_source = os.path.join(input_path, data_folder,
                                '<file name>.pickle')

## Read data frame from a '.pickle' file
## /!\ 'Pickle' is a binary format so reading pickle data from untrusted sources can be unsafe
df_git = pd.read_pickle(git_data_source)
df_jira = pd.read_pickle(jira_data_source)

## Get summary statistics for each data frame
git_stats = dataframe_summary (df_git)
jira_stats = dataframe_summary (df_jira)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## CLEAN DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
## Select the minimal set of relevant attributes per data set
git_relevant_features = [
    'issues',
    'commitDate',
    'filename',
    'authorName',
    'hash',
    'added',
    'deleted'
]

jira_relevant_features = [x for x in df_jira.columns if x not in []]

## Reduce data sets to the selected relevant features
df_git = df_git[git_relevant_features]
df_jira = df_jira[jira_relevant_features]

##-------------------------------------------------------------------------------------------------
## Renam feature for convenience (improved undestandability)
df_git.rename(columns = {
    'issues':       'Commit_Issue',
    'commitDate':   'Commit_Date',
    'filename':     'Commit_File',
    'authorName':   'Commit_Author',
    'hash':         'Commit_ID',
    'added':        'Commit_Added_LOC',
    'deleted':      'Commit_Deleted_LOC'
    }, inplace = True
)

## Check the number of unique issues in Issue-Sprint and Issue-Changes data sets
print('Jira unique issues count:', df_jira['Issue_ID'].nunique())

## Check whether Git-Issues contain issues that are not reported in Jira
jira_issues = df_jira['Issue_Key'].unique().tolist()
git_issues = df_git['Commit_Issue'].unique().tolist()
common_issues = list(set(jira_issues).intersection(set(git_issues)))
print ('Number of unique issue in Git:', len(git_issues))
print ('Number of unique issues in Jira;', len(jira_issues))
print('Number of common issues in Git and Jira:', len(common_issues))

##-------------------------------------------------------------------------------------------------
## Select RELEVANT data

## Remove entries for non-java files (generally for out-of-scope data file types)
#df_git = df_git[df_git['Commit_File'].str.endswith('.java', na = False)]

##-------------------------------------------------------------------------------------------------
## Handle MISSING VALUES

## Set how much missing can maximally remain per feature
max_missing = 0.2
missing_threshold = 1 - max_missing

## Identify features where %missing > 1-max_missing
print("Git features with more than", missing_threshold*100, "% missing entries:",
      df_git.columns[(df_git.isna().sum() / len(df_git)) > (missing_threshold)].tolist())
print("Jira features with more than", missing_threshold*100, "% missing entries:",
    df_jira.columns[(df_jira.isna().sum() / len(df_jira)) > (missing_threshold)].tolist())

## Handle missing issue-type ('issues', 'issuekey')
## OPTION 1: Replace missing issue IDs with a constant value (many messages suggest 'Refactoring')
df_git['Commit_Issue'].fillna('Unknown', inplace = True)

## OPTION 2: Remove rows/entries where required information on issue-type is missing
# df_git.dropna(axis=0, subset=['Commit_Issue'], inplace=True)

## Remove remaining features with majority of entries missing (if any such exists)
df_git.dropna(thresh=len(df_git) * max_missing, axis=1, inplace=True)
df_jira.dropna(thresh=len(df_jira) * max_missing, axis=1, inplace=True)

##-------------------------------------------------------------------------------------------------
## Handle DUPLICATES

## Git: Multiple issues of the same type in the same file but in different parts of code were
## corrected in the same commit
## Solution: Remove duplicates by aggregating 'added' and 'deleted' LOC withn each commit
aggregation_functions = {'Commit_Added_LOC':'sum',
                         'Commit_Deleted_LOC':'sum',
                         'Commit_Author':[mode, 'count']}
groupping_features = ['Commit_Issue', 'Commit_Date', 'Commit_ID', 'Commit_File']
#groupping_features = [x for x in df_git.columns if x not in list(aggregation_functions.keys())]
df_git = df_git.groupby(groupping_features, as_index = False).agg(aggregation_functions)
df_git.columns = df_git.columns.map(''.join)
df_git.rename(columns = {
                        'Commit_Added_LOCsum':      'Commit_Added_LOC',
                        'Commit_Deleted_LOCsum':    'Commit_Deleted_LOC',
                        'Commit_Authormode':        'Commit_Author',
                        'Commit_Authorcount':       'Commit_Issue_Count'
                       }, inplace = True)

## OPTIONALLY: Git: Mey still contain duplicates:
## the same time stamp, file, author, and even added and deleted LOC
## SOLUTION: Sort on Commit ID, File and Date, and select the most recent entry
# features_for_duplicates = ['Commit_Issue', 'Commit_File', 'Commit_Date']
# df_git.sort_values(features_for_duplicates, ascending=False, inplace=True)
# duplicates_git = df_git.groupby(
#     df_git[features_for_duplicates].columns.tolist()).size().reset_index()\
#     .rename(columns={0:'duplicates'})\
#     .sort_values('duplicates', ascending = False)
# print(duplicates_git.head())
#
# df_git.sort_values(features_for_duplicates, ascending = False, inplace = True)
# df_git.drop_duplicates(subset = features_for_duplicates, keep = 'first', inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## INTEGRATE Git with Jira (add issue infomration to commits
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Select final set relevant features (containing relevant information) to integrate
jira_features = ['Issue_Key', 'Issue_Type', 'Issue_Priority',
                 'Issue_Creation_Date', 'Issue_Component', 'Issue_Platform',
                 'Issue_Resolution_Date', 'Issue_Update_Date', 'Issue_Reporter',
                 'Issue_Assignee', 'Issue_Status']
git_features = ['Commit_Issue', 'Commit_Date', 'Commit_ID', 'Commit_File',
                'Commit_Added_LOC', 'Commit_Deleted_LOC', 'Commit_Author',
                'Commit_Issue_Count']

df_git_jira = pd.merge(left = df_git[git_features], right = df_jira[jira_features],
                       how = 'left', left_on='Commit_Issue', right_on='Issue_Key',
                       suffixes=('_git', '_jira'))

## Clean up data: remove unnecessary/redundant features
df_git_jira.drop('Commit_Issue', axis = 1, inplace = True)

## Get summary stats (custom-defined function in 'Functions.py')
#summary_stats = dataframe_summary(df_git_jira)

## Sort data
sorting_features = ['Commit_Date', 'Commit_File', 'Commit_ID']
df_git_jira.sort_values(sorting_features, ascending = False, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## SAVE data to file
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Save data to a '.pickle' file
data_target = os.path.join(output_path, 'PICKLE',
                           '<output file name>.pickle')
df_git_jira.to_pickle(data_target)

git_data_target = os.path.join(input_path, 'PICKLE',
                               '<output file name>.pickle')
df_git.to_pickle(git_data_target)

###################################################################################################
### EOF ###