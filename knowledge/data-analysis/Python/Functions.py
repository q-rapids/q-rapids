###################################################################################################
# PURPOSE:  Custom functions
#
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           numpy (np), pandas (pd), seaborn (sns), matplotlib.pyplot (plt), 
#           matplotlib.ticker (ticker), time, calendar, datetime.datetime, datetime.timedelta
#           Please read and comply to the applicable usage licenses.
####################################################################################################

##-------------------------------------------------------------------------------------------------
def dataframe_summary (df):
    """
    Computes summary statisitcs of a data frame

    :param df: Dataframe to comoute statistics for
    :return: Dataframe with summary statisitcs for input data frame
    """
    df_summary = df.describe(include='all')
    df_summary.loc['Missing #'] = df.isnull().sum(axis=0)
    df_summary.loc['Missing %'] = round(df.isnull().sum(axis=0) / len(df.index) * 100, 1)
    return df_summary

##-------------------------------------------------------------------------------------------------
## @title:     Sorts columns of a data frame on given criteria: Currently only #Missing
## @df:        Data frame to sort (returned after sorting)
## @criteria:  Sort criteria (currently #Missing per column
##@return:     Input data frame with columns sorted according to the 'Criteria'
def sort_dataframe_columns (df, criteria = 'Missing'):
    """
    Sorts columns of a data frame on given criteria: Currently only #Missing

    :param df:          Data frame to sort
    :param criteria:    Sort criteria (currently #Missing per column
    :return:            Input dataframe with columns sorted accroding to criteria
    """
    nans = pd.DataFrame (data = df.isnull().sum(axis = 0), columns = ['Missing'])
    nans = nans.sort_values('Missing', ascending = False)
    return df[nans.index.tolist()]

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## TIME
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
def elapset_time(start, end):
    """
    Compute difference in two dates in days:hours:minutes

    :param start:   Start date
    :param end:     End date
    :return:        Time dfference in seconds
    """
    diff = end - start
    seconds = diff / np.timedelta64(1, 's')
    #days = int(seconds // (24 * 3600))
    #seconds = seconds % (24 * 3600)
    #hours = int(seconds // 3600)
    #seconds %= 3600
    #minutes = int(seconds // 60)
    #seconds %= 60
    #seconds = int(seconds)

    return timedelta(seconds = seconds)
    #return timedelta(seconds = seconds, minutes = minutes, hours = hours, days = days)
    #return days, hours, minutes, seconds

##-------------------------------------------------------------------------------------------------
def derive_time_features(df, source_date_column = 'index'):
    """
    Creates time series features from datetime index

    :param df:                  Data frame representing time series
    :param source_date_column:  Column in df representing time
    :return:                    Data frame with derived features only
    """
    if (source_date_column == 'index'):
        df['date'] = df.index
        date_col = 'date'
    else:
        date_col = source_date_column

    #df['hour'] = df[date_col].dt.hour
    #df['hour'] = df[date_col].dt.minute
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['dayname'] = df['dayofweek'].apply(lambda x: calendar.day_name[x])
    df['quarter'] = df[date_col].dt.quarter
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['dayofmonth'] = df[date_col].dt.day
    df['weekofyear'] = df[date_col].dt.weekofyear

    X = df[[
            #'hour',
            #'minute',
            'dayofweek',
            'quarter',
            'month',
            'year',
            'dayofyear',
            'dayofmonth',
            'weekofyear'
           ]]
    return X

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## VISUALIZATION
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##-------------------------------------------------------------------------------------------------
def corr_heatmap (corr_matrix, font_scale = 1, plot_title = 'Heatmap'):
    """
    Wrapper of the 'Seaborn.heatmap' for visualizing symetrical matrix (e.g, distance,
    similarity, or correlation)

    :param corr_matrix: Data metric to visualize
    :param font_scale:  Scaling factor for font size on the heatmap
    :param plot_title:  Title to of the heatmap to be displayed
    :return:            seaborn as sns, matplotlib.pyplot as plt
    """
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True

    plt.figure()
    sns.set(font_scale = font_scale)
    with sns.axes_style("white"):
        ax = sns.heatmap(corr_matrix,
                         cmap = sns.color_palette("RdBu_r", 50),
                         xticklabels=corr_matrix.columns.values,
                         yticklabels=corr_matrix.columns.values,
                         mask=mask, square=True)
    ax.set_title(plot_title)
    plt.subplots_adjust(bottom = 0.3, left = 0.3)

    return ax.get_figure()

## def 'corr_heatmap'

##-------------------------------------------------------------------------------------------------
def countplot_wrapper (df, feature, sorting = [], plot_title = 'Histogrtam'):
    """
    Wrapper of the 'Seaborn.countplot' for visualizing histogram of categorical
    variable counts

    :param df:          Data frame
    :param feature:     Categorical feature to visualize
    :param sorting:     Reference list of strings according to which categories on the histogram
                        are to be sorted
    :param plot_title:  Title to of the heatmap to be displayed
    :return:            Reference list of strings according to which categories on the histogram
                        are to be sorted
    """
    ncount = len(df)
    ncolors =  len(df[feature].unique())

    palette = sns.husl_palette(ncolors, s = 0.7)

    ## If no reference sorting is given then sort unique values of the categorical feature
    ## according to their absolute frequency on data df
    if len(sorting) == 0:
        unique_values = pd.value_counts(df[feature]).sort_values(ascending = False)
        sorting = unique_values.index.tolist()

    plt.figure(figsize = (12, 8))
    ax = sns.countplot(x = feature, data = df, order = sorting, palette = palette)
    plt.title(plot_title)
    plt.xlabel('Week Day')

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')
    ax.set_ylabel('Absolute Count')

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),
                ha='center', va='bottom') # set the alignment of the text

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to 0-100
    ax2.set_ylim(0,100)
    ax.set_ylim(0,ncount)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    ax2.grid(None)

## def 'countplot_wrapper'

