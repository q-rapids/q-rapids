####################################################################################################
# PURPOSE: Transform raw data. Basic transformations:
# - Discard irrelevant attributes (e.g., ElasticSearch index descriptions)
# - Set proper attribute types (R imports all attributes as 'character' type)
# - Unfold data (measurement data are imported from ElasticSearch in folded format)
#
# WARNING:  Runnig this script reguires inclusion of the following external modules (libraries): 
#           reshape2, caret
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
dataSet <- 'sonar.measures.latest'
dataVersion <- '2019-05-06'
dataVariant <- 'raw'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Load data
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
df <- readRDS(file = paste0(inputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-', dataVariant, '.rds'))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# EXTRACT RELEVANT DATA: Create separate attributes for each metric of interest
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Remove none '_source' attributes and remove '_source' prefix from remaining attributes' names
df <- df[, grepl("*_source", names(df))]
names(df) <- gsub("_source.", "", names(df))

## Save intermediate outcome (raw data without ElasticSearch imformative attributes) to RDS
saveRDS(df, file = paste0(outputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-raw-baseline', '.rds'))

## Remove empty and constant-value attributes (e.g., all attribute's values either equal or NA)
df <- df[, which(lapply(df, function (x) length(unique(x))) > 1)]

## The data after extraction is melted and must be unmelted on metric-value attributes.
## Select metric attribute to pivoted the data set on
metricName <- 'metric'
metricValue <- 'value'

## Select relevant secondary attributes
contextMetrics <- c(
  'snapshotDate',
  'path',
  'language',
  'commit',
  'name',
  'Id'
)

## Filer out irrlevant rows and columns (ToDo: implement as functinon)
df <- df[df$qualifier %in% c("FIL") & df$language %in% c("java"), c(metricName, metricValue, contextMetrics)]

## Check if rows on ID attributes are unique (required for correct unmelting)
keyAttributes <- c('snapshotDate', 'path', 'commit', 'metric')#, 'Id')
#duplicatesAllCount <- nrow(df) - nrow(unique(df[, keyAttributes]))
duplicatedRowsAll <- df[duplicated(df[,keyAttributes]),]
duplicatedRowsUnique <- unique(duplicatedRowsAll)
duplicatesAllCount <- nrow(duplicatedRowsUnique)
duplicatesUniqueCount <- nrow(duplicatedRowsUnique)
cat("Total duplicates: ", duplicatesAllCount, "| Unique rows duplicated: ", duplicatesUniqueCount)

sortFeatures <- c('metric', 'snapshotDate', 'path', 'commit', 'Id')


duplicatedRowsUnique <- duplicatedRowsUnique[do.call("order", duplicatedRowsUnique[sortFeatures]), ]
duplicatedRowsAll <- duplicatedRowsAll[do.call("order", duplicatedRowsAll[sortFeatures]), ]

duplicates <- df[df$Id %in% duplicatedRowsAll$Id, ]
duplicates <- duplicates[do.call("order", duplicates[sortFeatures]), ]
write.table(duplicates, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', dataSet, '.', dataVersion, '-DUPLICATES', '.csv'), 
            sep = ";", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)

## Remove duplicates
df <- df[!duplicated(df[,keyAttributes]),]

## Define unmelting formula
unmeltFeatures <- c(
  'snapshotDate',
  'path',
  'commit'
)
unmeltFormula <- as.formula(paste(paste(unmeltFeatures, collapse = " + "), metricName, sep = " ~ "))

## Unmeld data (for each 'targetMetric' create a separate attribute)
df <- reshape2::dcast(df, unmeltFormula, value.var = metricValue)#, fun.aggregate = mean)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# SET DATA TYPE: Set the right type of each attribute (R imports any data as 'character' type)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Select attributes according to their type
## Float attributes:
numCols <- c(
  'comment_lines_density',
  'duplicated_lines_density'
)

## Integer attributes:
intCols <- c(
  'bugs',
  'classes',
  'code_smells',
  'cognitive_complexity',
  'comment_lines',
  'complexity',
  'duplicated_blocks',
  'duplicated_files',
  'duplicated_lines',
  'files',
  'functions',
  'lines',
  'ncloc',
  'new_bugs',
  'new_code_smells',
  'new_violations',
  'new_vulnerabilities',
  'reliability_rating',
  'security_rating',
  'statements'
)

## Logical attributes:
boolCols <- c(
)

## Date attributes:
dateCols <- c(
  "snapshotDate"
)

## Categorical (factor) attributes:
catCols <- c()

## Character attributes (all remaining attributes, not considerred in previous groups):
charCols <- setdiff(colnames(df), c(numCols, intCols, boolCols, dateCols, catCols))

## Transform attributes to their proper type 
## For date type, extract first 10 characters of year-month-day from the source 
## date entry (e.g., 2017-12-01T11:04:01.41) and convert it to date format of R
if(length(numCols) != 0) df[numCols] <- lapply(df[numCols], function(x) as.numeric(x))
if(length(intCols) != 0) df[intCols] <- lapply(df[intCols], function(x) as.integer(x))
if(length(boolCols) != 0) df[boolCols] <- lapply(df[boolCols], function(x) as.logical(x))
if(length(dateCols) != 0) df[dateCols] <- lapply(df[dateCols], function(x) as.POSIXct(x, format = "%Y-%m-%d", tz = Sys.timezone(location = TRUE)))
if(length(charCols) != 0) df[charCols] <- lapply(df[charCols], function(x) as.character(x))
if(length(catCols) != 0) df[catCols] <- lapply(df[catCols], function(x) as.factor(x))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# SAVE DATA: Unfolded & Formated
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Save unfolded data to file
## RDS file
saveRDS(df, file = paste0(outputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-unfolded-formatted', '.rds'))

## Feather file
install.packages('feather', repo='https://cran.cnr.Berkeley.edu/')
library (feather)
write_feather(df, path = paste0(outputPath, 'FEATHER/', dataSource, '.', dataSet, '.', dataVersion, '-unfolded-formatted', '.feather'))

## OPTIONALLY: CSV file (Warning: csv takes a lot of storage and can be tricky with special character, tabs and semi-commas)
write.table(df, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', dataSet, '.', dataVersion, '-unfolded-formatted', '.csv'), 
            sep = ";", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)



### EOF ###