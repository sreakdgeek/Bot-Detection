directory <- '/home/nivvi80/hackathon/bot_classify'
setwd(directory)

library(dplyr)

#Reading the datasets
bids <- read.csv('/home/nivvi80/hackathon/bot_classify/bids.csv')
bidder <- read.csv('train.csv')
dasrath_file <- read.csv('bidders_count_country_ip_devices.csv')

#Getting the outcome variable to the bid dataset
bids <- left_join(bids,bidder,by=NULL)

#Feature Engineering  - Finding some relevant features
#Hypo 1  - # devices / auction ID / bid ID
#Hypo 2 - # References per bidder_id per auction (URL)
device <- bids %>% group_by(bidder_id) %>% summarise(total_devices = n_distinct(device),
                                                             total_references = n_distinct(url),
                                                             total_auctions = n_distinct(auction)) %>% mutate(device_auction = total_devices/total_auctions,
                                                 references_auction = total_references/total_auctions)
#Hypo 3 - #Times the bidder is the last bidder for an auction
last_bid <- bids %>% select(bidder_id,auction,bid_id) %>% 
  group_by(auction) %>% summarise(bid_id = max(bid_id))
bids$last_bud_flag <- 0
bids$last_bud_flag[bids$bid_id %in% last_bid$bid_id] <- 1
bids[is.na(bids)] <- 0

#Hypo 4 - #Times Bid won - Last bid / Total_bids_placed
bid_won <- bids %>% group_by(bidder_id) %>% summarise(bids_won = sum(last_bud_flag),
                                                      bids_placed = length(bid_id)) %>% mutate(won_percentage = bids_won/bids_placed * 100)

master_bid_set <- left_join(bidder,device,by=NULL)
master_bid_set <- left_join(master_bid_set,bid_won[,c(1,4)],by=NULL)
master_bid_set <- left_join(master_bid_set,dasrath_file,by=NULL)
master_bid_set[is.na(master_bid_set)] <- 0
master_bid_set$country_flag <- 0
master_bid_set$country_flag[master_bid_set$multiple_country == 'true'] <- 1

#Creating the bids and ip per auction variables
master_bid_set$bid_auction <- master_bid_set$bid_count/master_bid_set$total_auctions
master_bid_set$ip_auction <- master_bid_set$unique_ips/master_bid_set$total_auctions

#Removing the extra columns and rearranging
drop_colnames <- c('multiple_country','unique_devices','payment_account','address','bid_count',
                   'unique_ips','total_devices','total_references','total_auctions')
master_bid_set <- master_bid_set[,!colnames(master_bid_set) %in% drop_colnames]
master_bid_set <- master_bid_set[,c(1,3:8,2)]

#Writing the final training dataset
write.csv(master_bid_set,'master_bid_set.csv',row.names=F)
# mer_try <- bids %>% group_by(bidder_id) %>% summarise(total_mer  = n_distinct(merchandise))

#Oversampling - Increase the weightage of the signal in the dataset
#Subsetting the dataset for all one's(response variable - outcome)
master_bid_set_os <- master_bid_set[master_bid_set$outcome == 1,]
#Random Sampling the dataset for all zeroes
master_bid_set_zero <- master_bid_set[sample(nrow(master_bid_set[master_bid_set$outcome == 0,]),1000),]


##Merging the above two datasets to get a final train dataset
train_final <- rbind(master_bid_set_os,master_bid_set_zero)
mylogit <- glm(outcome ~ device_auction + references_auction + won_percentage + country_flag + bid_auction
               + ip_auction, data = train_final , family='binomial')

#Building the test dataset
test_dataset <- read.csv('test.csv')
master_test_set <- left_join(test_dataset,device,by=NULL)
master_test_set <- left_join(master_test_set,bid_won[,c(1,4)],by=NULL)
master_test_set <- left_join(master_test_set,dasrath_file,by=NULL)
master_test_set[is.na(master_test_set)] <- 0
master_test_set$country_flag <- 0
master_test_set$country_flag[master_test_set$multiple_country == 'true'] <- 1

#Creating the bids and ip per auction variables
master_test_set$bid_auction <- master_test_set$bid_count/master_test_set$total_auctions
master_test_set$ip_auction <- master_test_set$unique_ips/master_test_set$total_auctions

#Removing the extra columns and rearranging
drop_colnames <- c('multiple_country','unique_devices','payment_account','address','bid_count',
                   'unique_ips','total_devices','total_references','total_auctions')
master_test_set <- master_test_set[,!colnames(master_test_set) %in% drop_colnames]
write.csv(master_test_set,'master_test_set.csv',row.names=F)

prediction_set <- data.frame(predict(mylogit, master_test_set, type='response'))

submission_dataset <- data.frame(master_test_set$bidder_id,prediction_set)
names(submission_dataset)[] <- c('bidder_id','predict_prob')
write.csv(submission_dataset,'submission.csv',row.names=F)
