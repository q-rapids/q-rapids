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
# Initiate global variables
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
inputPath <- paste0(getwd(), '/0_Ingestion/')
outputPath <- paste0(getwd(), '/0_Ingestion/')
dataSource <- 'IESE'
dataSet <- 'jira-sprint-issue' # 'jira-sprint-issue' | jira-issue-changes
dataVersion <- '2019-01-29'
dataVariant <- 'raw'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Load data
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
df <- readRDS(file = paste0(inputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-', dataVariant, '.rds'))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# EXTRACT & PREPROCESS RELEVANT DATA
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Remove none '_source' attributes and remove '_source' prefix from remaining attributes' names
df <- df[, grepl("*_source", names(df))]
names(df) <- gsub("_source.", "", names(df))

## Identify "empty" data entires, which despite diffrent codings indicate missing data.
empty_values <- sort(
                  unique(
                    unlist(
                      unname(
                        #lapply(df, function(x) x[!grepl("^[[:alpha:]]|^[[:digit:]]", x)])
                        lapply(df, function(x) x[grepl("^[^[:alpha:][:digit:]]*$", x)])
                      )
                    )
                  )
                )
## If any empty cell found then replace them with 'NA'
if (length(empty_values > 0)){
  ## Localize empty cells and create a logical mask
  mask <- t(apply(df, 1, function(x) x %in% empty_values))
  ## Replace identified empty cells with 'NA'
  df[mask] <- NA
}

## Code cells containing only empty list or empty character with 'NA'
for (col in colnames(df)){
  df[lengths(df[, col]) == 0, col] <- NA
}

## Remove empty and constant-value attributes (e.g., all attribute's values either equal or NA)
df <- df[, which(lapply(df, function (x) length(unique(x))) > 1)]
#df <- df[, -nearZeroVar(df, freqCut = 100, saveMetrics = FALSE, names = FALSE, foreach = FALSE, allowParallel = TRUE)]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# SET DATA TYPE: Set the right type of each attribute (R imports any data as 'character' type)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Select attributes according to their type
## Float attributes:
numCols <- c(
  'timespent',
  #'timeestimate', # jira-sprint-issue
  'aggregatetimespent',
  'aggregatetimeestimate'
)

## Integer attributes:
intCols <- c(
  #'sprinttotal'  # jira-sprint-issue
)

## Logical attributes:
boolCols <- c(
)

## Date-Time attributes:
dateTimeCols <- c(
  'resolutiondate',
  'lastViewed',
  #'created', # jira-sprint-issue
  'changedate', # jira-issue-chages
  'updated'
)

## Date (only date, without time) attributes:
dateCols <- c(
  'duedate'
)

## Categorical (factor) attributes:
catCols <- c()

## Character attributes (all remaining attributes, not considerred in previous groups):
charCols <- setdiff(colnames(df), c(numCols, intCols, boolCols, dateCols, dateTimeCols, catCols))

## Transform attributes to their proper type 
## For date type, use POSIXct or POSIXlt to import it correctly in python
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

## /!\ Text entries (e.g., git message) contain special characters (\n or ;), which lead to incorrect structure of CSV
write.table(df, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', dataSet, '.', dataVersion, '-formatted', '.csv'), 
            sep = ";", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)


#### EOF ####