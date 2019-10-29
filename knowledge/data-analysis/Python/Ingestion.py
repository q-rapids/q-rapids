###################################################################################################
# PURPOSE:  Ingestion of software quality data
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime.date, pandas (pd), numpy (np), seaborn (sns), matplotlib.pyplot (plt),
#           feather (ft), pyreadr, tzlocal
#           Please read and comply to the applicable usage licenses.
####################################################################################################

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## LOAD DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Set data storage path
input_path = os.path.normpath("<location/directory of input data>")
output_path = os.path.normpath("location/directory of output data ")
data_name = '<data file name>'
data_type = '.feather'
data_variant = '<data variant>'
data_version = '<data version time stamp>'

data_source = os.path.join(input_path, 'FEATHER', data_name + '.' + data_version + data_variant + data_type)

## Read feather data format (works also for large datasets where 'pyreadr' fails)
df = ft.read_dataframe(source = data_source)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## PREVIEW DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Check size of the data frame
df.shape
## View first rows of the data frame
df.head()
## Get data type of each column
df.dtypes

## Get summary statistics incl. missing data count and ratio
df_summary = dataframe_summary(df)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## VISUALIZE MISSING DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##--------------------------------------------------------------------
## Correlations between features w.r.t. missingness

## Calculate the correlation matrix
correlation_of_missing = df.isna().corr(method = 'spearman')

## Plot correaltion matrix as heatmap
#sns.heatmap(correlation_of_missing, cmap = sns.color_palette("YlOrRd", 30))
corr_heatmap (correlation_of_missing, title = 'Correlation of Missing Values')

##--------------------------------------------------------------------
## Compute ratios of missing values per feature (data frame column)
missing_per_feature = pd.DataFrame({'Missing Count': df.isna().sum(),
                                    'Missing Percent': (df.isna().sum() / len(df) * 100)})
missing_per_feature.sort_values('Missing Percent', ascending = False, inplace = True)
missing_per_feature.head()

## Set up wider bottom margin for long feature names
plt.subplots_adjust(bottom = 0.25)

## Plot bar char of missing data percentages
bp = sns.barplot(x = missing_per_feature.index, y = 'Missing Percent', data = missing_per_feature)
for item in bp.get_xticklabels():
    item.set_rotation(90)

##--------------------------------------------------------------------
## Compute ratios of missing values per item (data frame row)
missing_per_item = pd.DataFrame({'Missing Count': df.isna().sum(axis = 1),
                                 'Missing Percent': (df.isna().sum(axis = 1) / df.shape[1] * 100)})
missing_per_item.sort_values('Missing Percent', ascending = False, inplace = True)
missing_per_item.head()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HANDLE MISSING DATA: LIST-WISE DELETION
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##--------------------------------------------------------------------
## Remove features where %NA > Threshold
features_to_remove = df.columns[(df.isna().sum() / len(df)) > 0.8].tolist()
df.drop(features_to_remove, axis = 1, inplace = True)

##--------------------------------------------------------------------
## OPTIONAL: Remove items where remaining %NA > Threshold
## But first recompute missing values ratios per item (after some features being dropped before)
items_to_remove = df.index[(df.isna().sum(axis = 1) / df.shape[1]) > 0.8].tolist()
df.drop(items_to_remove, axis = 0, inplace = True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HANDLE MISSING DATA: IMPUTATION
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## OPTIONAL: Since df represents time series, missing values can be interpolated using adjacent values
## Interpolate all missing values (do it for features that contain at least on 'NaN' value)
features_to_impute = df.columns[df.isna().any()].tolist()
df[features_to_impute] = df[features_to_impute].apply(lambda x: x.interpolate(method = 'linear'))

## ToDo: OPTIONAL: Handling categorical features needs to be implemented separately, if necessary

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## RENAME FEATURES
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Rename for SONAR-METRICS
df.rename(columns={'snapshotDate':                'Date',
                   'path':                        'File',
                   'commit':                      'Commit_ID',
                   'name':                        'File_Name',
                   'bugs':                        'Bugs',
                   'classes':                     'Classes',
                   'code_smells':                 'Code_Smells',
                   'cognitive_complexity':        'Cognitive_Cplx',
                   'comment_lines':               'Comment_Lines',
                   'comment_lines_density':       'Comment_Density',
                   'complexity':                  'Complexity',
                   'duplicated_blocks':           'Duplicate_Blocks',
                   'duplicated_files':            'Duplicate_Files',
                   'duplicated_lines':            'Duplicate_Lines',
                   'duplicated_lines_density':    'Duplicate_Lines Density',
                   'files':                       'Files',
                   'functions':                   'Functions',
                   'lines':                       'Total_LOC',
                   'ncloc':                       'Non_Comment_LOC',
                   'reliability_rating':          'Reliability',
                   'security_rating':             'Security',
                   'statements':                  'Statements'
                  }, inplace = True)

## Rename for SONAR-ISSUES
df.rename(columns={'snapshotDate':  'Date',
                   'component':     'File',
                   'commit':        'Commit_ID',
                   'severity':      'Severity',
                   'line':          'Line',
                   'author':        'Author',
                   'rule':          'Rule',
                   'effort':        'Effort',
                   'message':       'Message',
                   'creationDate':  'Creation_Date',
                   'debt':          'Debt',
                   'key':           'Key',
                   'status':        'Status',
                   'endLine':       'End_Line',
                   'endOffset':     'End_Offset',
                   'startLine':     'Start_Line',
                   'startOffset':   'Start_Offset',
                   'severityCount': 'Severity_Count',
                   'linesCount':    'Lines_Count'
                   }, inplace=True)

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## STORE DATA
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Set data storage path
data_variant = "-prepared"
#data_version = date.today().strftime('%Y%m%d')
data_type = ".pickle"
data_target = os.path.join(input_path, 'PICKLE', data_name + '.' + data_version + data_variant + data_type)

## Save data to a '.pickle' file
df.to_pickle(data_target)

###################################################################################################
### EOF ###