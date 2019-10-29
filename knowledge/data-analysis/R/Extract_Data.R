####################################################################################################
# PURPOSE:  Extract Q-Rapids software quality data from Elasticsearch
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

## Data locations in ElasticSearch
dfDataSources <- data.frame(Location = c('10.128.13.182', #'10.128.13.182'| '10.128.13.181', 
                                      '10.128.13.180', 
                                      '10.128.13.183', 
                                      '10.128.13.182', 
                                      '193.142.112.120'), 
                         row.names = c('IESE', 
                                       'Bittium', 
                                       'Softeam', 
                                       'Nokia', 
                                       'ITTI'),
                         stringsAsFactors = FALSE)
dataSource <- 'IESE'
dataVariant <- 'raw'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
# Extract data from ElasticSearch and load into a data frame
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

## Connect to ElasticSearch 
connect(es_host = dfDataSources[dataSource, 'Location'], es_port = 9200)

## Display connection detailes
info()

## Get infomration on all existing indices (name, documents count, storage size)
esIndices <- cat_indices(verbose = TRUE,  h = c('index','docs.count','store.size'), parse = TRUE)
print (esIndices)

## Set pattern for extracting the field name from a mapping information
pattern <- "properties\\.(.*?)\\."

## Select which indices should be considered
esIndicesFirst <- 1
esIndicesLast <- nrow(esIndices)

## Get field names for each i-th index
for (i in esIndicesFirst:esIndicesLast){
  ## Select specific index
  esIndex <- esIndices$index[i]
  
  ## Get informaiton on the intext mappings
  esIndexMappings <- mapping_get(index = esIndex)
  
  ## Extract names of fields in the index
  #esIndexFields <- names(esIndexMappings[[1]][[1]][[1]][[1]])
  esIndexFields <- names(unlist(esIndexMappings))
  
  if (length(esIndexFields) != 0){
    esIndexFields <- unique(
      lapply(esIndexFields, function (x) regmatches(x,regexec(pattern,x))[[1]][2]))
    esIndexFields <- esIndexFields[!is.na(esIndexFields)]
    
    print(paste0("### INDEX: ", esIndex," ###"))
    print(paste0(esIndexFields))
    cat("\n\n")
  }
  
}

## Set up scrolling parameters to search through the given index
## Relevant Indices in the context of 'Digitale Doerfer' project:
##   git: commit history
##   jira-sprint-issue: issue tracking (but only initial state of issues)
##   jira-issue-changes: changes of issues state (e.g., 'open' -> 'closed')
##   sonar.metrics.new: SonarQube metrics
##   sonar.issues.new: SonarQube issues
##   hockeyapp1: Runtime data for the 'Hockeyapp' application
esIndex <- 'sonar.issues.latest'
esHitsCount <- '10000'
esScrollTime <- '1m'
esRecordsCount <- Search(index = esIndex, size = 0)$hits$total

## Get data using Scroll and load into DATA FRAME
## /!\ Initial search already returnds the first part of data of size defined within the seach function
##     The following scroll returns data parts beginning from size+1 element.
res <- Search(index = esIndex, time_scroll = esScrollTime, size = esHitsCount, body = '{"sort": ["_doc"]}', asdf = TRUE)
out <- res$hits$hits 
print(nrow(out)) # current number of hits loaded into the dataframe
hits <- 1
while(hits != 0){
  res <- scroll(x = res$`_scroll_id`, asdf = TRUE)
  hits <- length(res$hits$hits)
  #print(hits)
  if(hits > 0)
    out <- dplyr::bind_rows(out, res$hits$hits)
  print(nrow(out)) # current number of hits loaded into the dataframe
}

## Save imported RAW DATA to RDS and CSV files
saveRDS(out, file = paste0(outputPath, 'RDS/', dataSource, '.', esIndex, '.', Sys.Date(), '-', dataVariant,'.rds'))

## Flatten columns of type 'list' produced by extracting data from ElasticSearch
## Write.table cannot handle data frame columns of type 'list'
for (cn in colnames(out))
  if (is.list(out[, cn]))
  {
    #print(paste0(cn, '|', class(out[, cn])))
    out[, cn] <- sapply(out[, cn], function(x) paste(unlist(x), collapse = ";"))
    out[, cn] <- as.character(out[, cn])
  }

write.table(out, 
            file = paste0(outputPath, 'CSV/', dataSource, '.', esIndex, '.', Sys.Date(), '-', dataVariant,'.csv'), 
            sep = ",", 
            dec = ".", 
            na = "",
            row.names = FALSE,
            col.names = TRUE)


### EOF ###