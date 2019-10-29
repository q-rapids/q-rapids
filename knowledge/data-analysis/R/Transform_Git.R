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
dataSet <- 'git'
dataVersion <- '2019-05-02'
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

## Save intermediate outcome (raw data without ElasticSearch imformative attributes) to RDS
saveRDS(df, file = paste0(outputPath, 'RDS/', dataSource, '.', dataSet, '.', dataVersion, '-raw-baseline', '.rds'))

## Identify "empty" data entires, which despite diffrent codings indicate missing data.
sort(
  unique(
    unlist(
      unname(
        #lapply(df, function(x) x[!grepl("^[[:alpha:]]|^[[:digit:]]", x)])
        lapply(df, function(x) x[grepl("^[^[:alpha:][:digit:]]*$", x)])
      )
    )
  )
)

## Code cells containing only empty list or empty character with 'NA'
for (col in colnames(df)){
  df[lengths(df[, col]) == 0, col] <- NA_character_
}

## Remove empty and constant-value attributes (e.g., all attribute's values either equal or NA)
df <- df[, which(lapply(df, function (x) length(unique(x))) > 1)]
#df <- df[, -nearZeroVar(df, freqCut = 100, saveMetrics = FALSE, names = FALSE, foreach = FALSE, allowParallel = TRUE)]

## Resolve entries where a single change handled more than one issue
## Solution: For each entry with N issues create N entries, one per eachh issue;
##           distribute 'added' and 'deleted' LOC proportionaly, copy other attributes
nested_col <- 'issues'
id_col <- c('commitDate', 'authorDate', 'filename', 'hash', 'message', 'authorName', 'authorEmail', 'committerName', 'committerEmail')
num_col <- c('added', 'deleted')

## Set type of numeric columns to avoid errors (all columns are initially of type 'character')
df[num_col] <- lapply(df[num_col], function(x) as.numeric(x))

col_names <- c(nested_col, id_col, num_col)

## Create an empty target data frame to filled with unnested data derived from the source data frame
df_target <- data.frame(matrix(ncol = length(col_names), nrow = 0), stringsAsFactors = FALSE)
colnames(df_target) <- col_names

## Go through all rows of the data frame
for (i in 1:nrow(df)) {
  ## For rows with multiple elemens in the 'nested_col' cell derive one row for each element
  if (!is.na (df[i, nested_col]) & length(df[i, nested_col][[1]]) > 1){
    #print(df[i, nested_col])
    
    ## Get the number of elements in the nested cell
    len <- length(df[i, nested_col][[1]])
    
    ## Derive new row for each cell (values of the numeric cells are distributed proportionally)
    unnested_rows <- unnest(df[i, col_names])
    unnested_rows[, num_col] <- round(unnested_rows[, num_col] / 2, 0)
    df_target <- rbind(df_target, unnested_rows)
    
  } else { ## Row with 0 or 1 element ist just copied to target data frame
    df_target <- rbind(df_target, df[i, col_names])
  }
}

## Remove nested data frame
df <- df_target
rm(df_target)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# SET DATA TYPE: Set the right type of each attribute (R imports any data as 'character' type)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Select attributes according to their type
## Float attributes:
numCols <- c(
  
)

## Integer attributes:
intCols <- c(
  'added',
  'deleted'
)

## Logical attributes:
boolCols <- c(
)

## Date attributes:
dateCols <- c(
  'commitDate',
  'authorDate'
)

## Categorical (factor) attributes:
catCols <- c()

## Character attributes (all remaining attributes, not considerred in previous groups):
charCols <- setdiff(colnames(df), c(numCols, intCols, boolCols, dateCols, catCols))

## Transform attributes to their proper type 
## For date type, use POSIXct or POSIXlt to import it correctly in python
if(length(numCols) != 0) df[numCols] <- lapply(df[numCols], function(x) as.numeric(x))
if(length(intCols) != 0) df[intCols] <- lapply(df[intCols], function(x) as.integer(x))
if(length(boolCols) != 0) df[boolCols] <- lapply(df[boolCols], function(x) as.logical(x))
#if(length(dateCols) != 0) df[dateCols] <- lapply(df[dateCols], function(x) as.Date(x, format = "%Y-%m-%d"))
if(length(dateCols) != 0) df[dateCols] <- lapply(df[dateCols], function(x) as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%S", tz = Sys.timezone(location = TRUE)))
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



### EOF ###


