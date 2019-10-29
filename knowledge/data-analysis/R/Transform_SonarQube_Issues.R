####################################################################################################
# PURPOSE: Transform raw data. Basic transformations:
# - Discard irrelevant attributes (e.g., ElasticSearch index descriptions)
# - Set proper attribute types (R imports all attributes as 'character' type)
# - Unfold data (measurement data are imported from ElasticSearch in folded format)
#
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           reshape2, tidyr, stringi, caret
#           Please read and comply to the applicable usage licenses.
####################################################################################################

## Set working directry
setwd("<path to working directory>")

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Custom Functions
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#' @title Convert time string to numeric time in hours 
#' @description Converts a string in a form '[[:digit:]]h[[:digit:]]min' into numeric hours
#' @param timeString String desribing tim in hours and minutes
#' @return Numeric time in hours
#' 
StringToTime <- function (timeString) {
  if (is.na(timeString) | length(timeString) == 0)
    return (NA)
  
  hourMinuteSplit <- unlist(strsplit(gsub(timeString, pattern = "min", replacement = ""),
                                     split = 'h'))
  
  if (length(hourMinuteSplit) > 1) {
    hours <- as.numeric(hourMinuteSplit[1])
    minutes <- as.numeric(hourMinuteSplit[2]) / 60
    
  } else {
    hours <- 0.0
    minutes <- as.numeric(hourMinuteSplit[1]) / 60
  }
  
  return (round(hours + minutes, 2))
}


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Initiate global variables
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
inputPath <- paste0(getwd(), '/0_Ingestion/')
outputPath <- paste0(getwd(), '/0_Ingestion/')
dataSource <- 'IESE'
dataSet <- 'sonar.issues.latest'
dataVersion <- '2019-06-07'
dataVariant <- 'raw'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Load data
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
df <- readRDS(file = paste0(inputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-', dataVariant, '.rds'))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# EXTRACT RELEVANT DATA
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Remove none '_source' attributes and remove '_source' prefix from remaining attributes' names
df <- df[, grepl("*_source", names(df))]
names(df) <- gsub("_source.", "", names(df))

## Remove empty and constant-value attributes (e.g., all attribute's values either equal or NA)
df <- df[, which(lapply(df, function (x) length(unique(x))) > 1)]
#df <- df[, -nearZeroVar(df, freqCut = 100, saveMetrics = FALSE, names = FALSE, foreach = FALSE, allowParallel = TRUE)]

## Transform attributes, e.g., effort 4h15min => 4x60+15 [min] OR 4+15/60 [h]
## Select time columns to convert
timeColumns <- c("effort", "debt")

## Run conversion function for all selected time columns
for (timeCol in timeColumns) {
  df[, timeCol] <- as.data.frame(unlist(lapply(df[, timeCol], function(x) StringToTime(x)))) 
}

## Check if there is 1:1 relationship between rule and severity (outcome must be 'TRUE')
## -> the same rule cannot have various severity levels
nrow(unique(df[, c("rule", "severity")])) == length(unique(df$rule))

## Add artificial attribute to count issues per group during the aggregation 
df$severityCount <- rep(1, nrow(df))

## For each issue count lines afected by the issue (lines between startLine and endLine)
df$linesCount <- ifelse(!(is.na(df$startLine) | is.na(df$endLine)), df$endLine - df$startLine + 1, NA)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# SET DATA TYPE: Set the right type of each attribute (R imports any data as 'character' type)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
## Select attributes according to their type
## Float attributes:
numCols <- c(
  'effort',
  'debt'
)

## Integer attributes:
intCols <- c(
  'endLine',
  'endOffset',
  'line',
  'startLine',
  'startOffset',
  'severityCount',
  'linesCount'
)

## Logical attributes:
boolCols <- c(
)

## Date-Time attributes:
dateTimeCols <- c(
  'creationDate'
)

## Date attributes:
dateCols <- c(
  'snapshotDate'
)

## Categorical (factor) attributes:
catCols <- c()

## Character attributes (all remaining attributes, not considerred in previous groups):
charCols <- setdiff(colnames(df), c(numCols, intCols, boolCols, dateCols, dateTimeCols, catCols))

## Transform attributes to their proper type 
## For date type, extract first 10 characters of year-month-day from the source 
## date entry (e.g., 2017-12-01T11:04:01.41) and convert it to date format of R
if(length(numCols) != 0) df[numCols] <- lapply(df[numCols], function(x) as.numeric(x))
if(length(intCols) != 0) df[intCols] <- lapply(df[intCols], function(x) as.integer(x))
if(length(boolCols) != 0) df[boolCols] <- lapply(df[boolCols], function(x) as.logical(x))
if(length(dateCols) != 0) df[dateCols] <- lapply(df[dateCols], function(x) as.POSIXct(x, format = "%Y-%m-%d", tz = Sys.timezone(location = TRUE)))
if(length(dateTimeCols) != 0) df[dateTimeCols] <- lapply(df[dateTimeCols], function(x) as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%S", tz = Sys.timezone(location = TRUE)))
if(length(charCols) != 0) df[charCols] <- lapply(df[charCols], function(x) as.character(x))
if(length(catCols) != 0) df[catCols] <- lapply(df[catCols], function(x) as.factor(x))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# SAVE DATA: Unfolded & Formated
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Save formated data to an RDS and CSV files
saveRDS(df, file = paste0(outputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-formatted', '.rds'))

## Feather file
install.packages('feather', repo='https://cran.cnr.Berkeley.edu/')
library (feather)
write_feather(df, path = paste0(outputPath, 'FEATHER/', dataSource, '.', dataSet, '.', dataVersion, '-formatted', '.feather'))

write.table(df, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', dataSet, '.', dataVersion, '-formatted', '.csv'), 
            sep = ";", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Derive new features/data
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Aggregate data on 'Date', 'FileName' and 'IssueSeverity' attributes
## use cbind(sum(x), mean(x), ...) to compute multiple functions for each attribte
dfAggr <- aggregate(formula = cbind(effort, debt, linesCount, severityCount) ~ snapshotDate + component + severity, data = df, FUN = function(x) sum(x))
dfAggr <- dfAggr[with(dfAggr, order(snapshotDate)), ]

## OPTIONAL: Unfold (break down) selected aggregated attribute per severity level (replace NA <- 0)
## To break down several attributes per severity the procedure must be repeated for each attribute
## and the outcomes must be combined (cbind)
dfUnfold <- reshape2::dcast(dfAggr, snapshotDate + component ~ severity, value.var = 'severityCount')
for(j in 1:ncol(dfUnfold))
  if (is.numeric(dfUnfold[[j]]))
    dfUnfold[[j]][is.na(dfUnfold[[j]])] <- 0

## Save data aggregated per Date-Component-Severity to an RDS and CSV files
saveRDS(dfAggr, file = paste0(outputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-formatted-aggregated', '.rds'))

write.table(dfAggr, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', dataSet, '.', dataVersion, '-formatted-aggregated', '.csv'), 
            sep = ";", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)

## Save data unfolded per each severity class to an RDS and CSV files
saveRDS(dfUnfold, file = paste0(outputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-formatted-unfolded', '.rds'))

write.table(dfUnfold, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', dataSet, '.', dataVersion, '-formatted-unfolded', '.csv'), 
            sep = ";", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)

### EOF ###