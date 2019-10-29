###################################################################################################
# PURPOSE:  Custom visualization functions
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           os, datetime.date, pandas (pd), numpy (np), seaborn (sns), matplotlib.pyplot (plt),
#           matplotlib.ticker (ticker), sklearn.utils.multiclass.unique_labels
#           sklearn.metrics.confusion_matrix
#           Please read and comply to the applicable usage licenses.
####################################################################################################

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## HEATMAP for CORRELATION MATRIX
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

## Visualize Correlations: Wrapper of the 'Seaborn.heatmap'
## Requires: seaborn as sns, matplotlib.pyplot as plt
def corr_heatmap (corr_matrix, font_scale = 1, title = 'Heatmap'):

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
    ax.set_title(title)
    plt.subplots_adjust(bottom = 0.3, left = 0.3)

    return ax.get_figure()

## def 'corr_heatmap'

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## COUNT PLOT with Ccounts and frequencies
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def count_plot (data, feature = '', title = '', label = '', annotation = 'count'):
    """
    Create count plot for unique values of a series (column of a data frame).

    :param data: Inout data frame
    :param feature: feature (column) in the data to visualize
    :param title: Plot title
    :param label: X-axis label
    :param annotation: Type of annotation at bars:
           'count': absolute number of observations with a given value,
           'frequency': percentage of observations with a given value
    :return: Plot
    """

    if title == '': title = "Frequency Plot"
    if label == '': label = 'Categories'

    ncount = len(data)
    plt.figure(figsize=(12, 8))

    if feature == '':
        categories = np.sort(data.unique())
        ax = sns.countplot(x = data, order = categories)
    else:
        categories = np.sort(data[feature].unique())
        ax = sns.countplot(x = feature, data = data, order = categories)

    plt.title(title)
    plt.xlabel(label)

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax.set_ylabel('Count')
    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x = p.get_bbox().get_points()[:,0]
        y = p.get_bbox().get_points()[1,1]
        if (annotation == 'count'):
            ax.annotate('{:d}'.format(int(y)), (x.mean(), y),
                        ha='center', va='bottom')  # set the alignment of the text
        if (annotation == 'frequency'):
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

    return plt

## def 'count_plot'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Plot Bar Chart: Individual importances
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def plot_feature_weights(feature_importances, orientation):
        x_values = list(range(len(feature_importances)))
        plt.barh(x_values, feature_importances['importance'],
                orientation = orientation, color = 'r', edgecolor = 'k', linewidth = 1.2)
        # Tick labels for x axis
        plt.xticks(x_values, feature_importances.index)#, rotation=orientation)
        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')

        return plt
# def 'plot_feature_weights'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Plot Line Chart: Cumulative importances
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plot_cumulative_feature_weights(feature_importances, orientation, importance_threshold = 0.75):
        # Compute cumulative importances
        cumulative_importances = np.cumsum(feature_importances['importance'])
        x_values = list(range(len(feature_importances)))
        # Make a line graph
        plt.plot(x_values, cumulative_importances, 'g-')
        # Draw line at 75% of importance retained
        plt.hlines(y = importance_threshold, xmin=0, xmax=len(feature_importances), color = 'r', linestyles = 'dashed')
        # Format x ticks and labels
        plt.xticks(x_values, feature_importances.index, rotation = orientation)
        # Axis labels and title
        plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')

        return plt
# def 'plot_cumulative_feature_weights'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Plot Confusion Matrix
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

## END of 'plot_confusion_matrix'

