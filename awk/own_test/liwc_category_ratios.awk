#!/usr/bin/gawk

#THIS SCRIPT IS WIRTTEN BASED ON MAX ___ AWK SCRIPT AS CAN BE FOUND
#IT OUTPUTS A RATIO OF LIWC CATEGORIES WORDS FOR EACH MESSAGE IN THE INPUT FILE

#################### BEGIN BLOCK ####################
#set field & record separator for inut and output file; initialize maxchar and minchar for later use
BEGIN {
    FS = " "; 
    ORS = "\n"; 
    OFS = ","; 
    maxchar = 0; 
    minchar = 100
}
#################### END OF BEGIN BLOCK ####################

#################### MAIN BLOCK ####################
{
    ########## THIS PART IS ONLY DONE ON FIRST INPUT FILE AKA DICTIONARY ##########
    #NR==FNR is an awk trick to make sure that we stay in the first file; here, our dictionary file
    if(NR == FNR) {
        category = $2 # get category
        numofcategory = $3 # get numofcategory
        
        # if first field contains a wildcard at the end, we do special substring matching later
        if($1 ~ /\*$/) {
            word = tolower(substr($1, 1, length($1) - 1)) # remove the wildcard at the end of the word
            pattern[word][numofcategory]
        } else {
            word = tolower($1)
            pattern_exact[word][numofcategory]
        }
        
        categories[numofcategory] = category # store the category with its number
        
        if(length($1) > maxchar) { # update maxchar if length of word is greater than current maxchar
            maxchar = length($1)
        }
        
        if(length($1) < minchar) { # update minchar if length of word is less than current minchar
            minchar = length($1)
        }
        
        #for the first file, we just build our lookup table and forget about the stuff below
        next
    }
########## END OF CODE FOR FIRST INPUT FILE AKA DICTIONARY ##########

########## THIS PART IS DONE ON THE SECOND INPUT FILE ##########
    #check if first record of current file
    if(FNR == 1) {
        header = "number_tokens"
        for(catnum in categories) {
            header = "liwc_" categories[catnum] "\t" header # construct header with liwc_ prefix
        }
        printf("%s\n", header) # print the header
    }

    #initialize hit counts for all categories to 0
    for(catnum in categories) {
        sum_hits[catnum] = 0
    }

    #loop over each field (one field = one token)
    for(i = 1; i <= NF; i++) {
        word = tolower($i) 
        
        #if there is an exact match, count 1 for each category the word belongs to
        if(word in pattern_exact) {
            for(numofcategory in pattern_exact[word]) {
                sum_hits[numofcategory]++
            }
            continue
        }
        
        #if there is no exact match, reduce the string and try to match it to the wildcard words
        for(s = maxchar; s >= minchar; s--) {
            subword = substr(word, 1, s)
            if(subword in pattern) {
                for(numofcategory in pattern[subword]) {
                    sum_hits[numofcategory]++
                }
                break
            }
        }
    }

    #prepare the result string with counts normalized by the number of tokens
    result = ""
    for(catnum in categories) {
        if (NF == 0) {
            result = result "0" "\t"
        } else {
            result = result (sum_hits[catnum] / NF) "\t"
        }
    }
    print result NF
    printf "\n"
}
#################### END OF MAIN BLOCK ####################