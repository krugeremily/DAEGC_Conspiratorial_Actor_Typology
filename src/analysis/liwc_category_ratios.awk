#!/usr/bin/gawk

# THIS SCRIPT IS WRITTEN BASED ON MAX ____ AWK SCRIPT AS CAN BE FOUND ___
# IT OUTPUTS A RATIO OF LIWC CATEGORIES WORDS FOR EACH MESSAGE IN THE INPUT FILE
# AS INPUT, IT TAKES A DICTIONARY FILE AND A FILE TO BE CLASSIFIED; THE LATTER SHOULD BE OF FORMAT ID WORD1 WORD2 WORD3 ...

#################### BEGIN BLOCK ####################
# set field & record separator for input and output file; initialize maxchar and minchar for later use
BEGIN {
    FS = " "; 
    ORS = "\n"; 
    OFS = ","; 
    maxchar = 0; 
    minchar = 100
    
    # define categories to classify
    # target_categories["Achieve"]=1
    # target_categories["Affect"]=1
    # target_categories["Anger"]=1
    # target_categories["Anx"]=1
    # target_categories["Article"]=1
    # target_categories["Assent"]=1
    # target_categories["Body"]=1
    # target_categories["Cause"]=1
    # target_categories["Certain"]=1
    # target_categories["Cogmech"]=1
    # target_categories["Comm"]=1
    # target_categories["Death"]=1
    # target_categories["Discrep"]=1
    # target_categories["Down"]=1
    # target_categories["Eating"]=1
    # target_categories["Excl"]=1
    # target_categories["Family"]=1
    # target_categories["Feel"]=1
    # target_categories["Fillers"]=1
    # target_categories["Friends"]=1
    # target_categories["Future"]=1
    # target_categories["Groom"]=1
    # target_categories["Hear"]=1
    # target_categories["Home"]=1
    # target_categories["Humans"]=1
    target_categories["I"]=1
    # target_categories["Incl"]=1
    # target_categories["Inhib"]=1
    # target_categories["Insight"]=1
    # target_categories["Job"]=1
    # target_categories["Leisure"]=1
    # target_categories["Metaph"]=1
    # target_categories["Money"]=1
    # target_categories["Motion"]=1
    # target_categories["Music"]=1
    # target_categories["Negate"]=1
    # target_categories["Negemo"]=1
    # target_categories["Nonfl"]=1
    # target_categories["Number"]=1
    # target_categories["Occup"]=1
    # target_categories["Optim"]=1
    target_categories["Other"]=1
    # target_categories["Othref"]=1
    # target_categories["Past"]=1
    # target_categories["Physcal"]=1
    # target_categories["Posemo"]=1
    # target_categories["Posfeel"]=1
    # target_categories["Preps"]=1
    # target_categories["Present"]=1
    # target_categories["Pronoun"]=1
    # target_categories["Relig"]=1
    # target_categories["Sad"]=1
    # target_categories["School"]=1
    # target_categories["See"]=1
    # target_categories["Self"]=1
    # target_categories["Senses"]=1
    # target_categories["Sexual"]=1
    # target_categories["Sleep"]=1
    # target_categories["Social"]=1
    # target_categories["Space"]=1
    # target_categories["Sports"]=1
    # target_categories["Swear"]=1
    # target_categories["TV"]=1
    # target_categories["Tentat"]=1
    # target_categories["Time"]=1
    # target_categories["Up"]=1
    target_categories["We"]=1
    target_categories["You"]=1  
}

#################### END OF BEGIN BLOCK ####################

#################### MAIN BLOCK ####################
{
    ########## THIS PART IS ONLY DONE ON FIRST INPUT FILE AKA DICTIONARY ##########
    # NR == FNR makes sure to stay in first input file aka dictionary
    if (NR == FNR) {
        category = $2 # get category
        numofcategory = $3 # get numofcategory
        
        # if first field contains a wildcard at the end, we do special substring matching later
        if ($1 ~ /\*$/) {
            word = tolower(substr($1, 1, length($1) - 1)) # remove the wildcard at the end of the word
            pattern[word][numofcategory]
        } else {
            word = tolower($1)
            pattern_exact[word][numofcategory]
        }

        # filter categories to only contain target categories
        if (category in target_categories) {
            categories[numofcategory] = category
        }
        
        if (length($1) > maxchar) { # update maxchar if length of word is greater than current maxchar
            maxchar = length($1)
        }
        
        if (length($1) < minchar) { # update minchar if length of word is less than current minchar
            minchar = length($1)
        }
        
        # for the first file, we just build our lookup table and forget about the stuff below
        next
    }
########## END OF CODE FOR FIRST INPUT FILE AKA DICTIONARY ##########

########## THIS PART IS DONE ON THE SECOND INPUT FILE ##########
    # skip the first field (unique ID) for classification purposes
    id = $1 # store the unique ID

    # check if first record of current file to print header
    if (FNR == 1) {
        header = "UID_key";

        # Construct header with LIWC categories
        for (catnum in categories) {
            if (categories[catnum] in target_categories) {
                header = header OFS "liwc_" categories[catnum];
            }
        }

        # Add a comma at the end of the header
        header = header OFS;

        print header;  # print the header
    }

    # initialize hit counts for all target categories to 0
    for (catnum in categories) {
        if (categories[catnum] in target_categories) {
            sum_hits[catnum] = 0
        }
    }

    # loop over each field (one field = one token), starting from the second field
    for (i = 2; i <= NF; i++) {
        word = tolower($i) 
        
        # if there is an exact match, count 1 for each category the word belongs to
        if (word in pattern_exact) {
            for (numofcategory in pattern_exact[word]) {
                if (categories[numofcategory] in target_categories) {
                    sum_hits[numofcategory]++
                }
            }
            continue
        }
        
        # if there is no exact match, reduce the string and try to match it to the wildcard words
        for (s = maxchar; s >= minchar; s--) {
            subword = substr(word, 1, s)
            if (subword in pattern) {
                for (numofcategory in pattern[subword]) {
                    if (categories[numofcategory] in target_categories) {
                        sum_hits[numofcategory]++
                    }
                }
                break
            }
        }
    }
## USE THIS PART FOR RATIOS ###
    # prepare the result string with counts normalized by the number of tokens
    result = id OFS
    for (catnum in categories) {
        if (categories[catnum] in target_categories) {
            if (NF == 1) { # handle case where there are no message tokens
                result = result "0" OFS
            } else {
                result = result (sum_hits[catnum] / (NF - 1)) OFS # normalize by the number of message tokens
            }
        }
    }
    print result # print the result
}

# ### USE THIS PART FOR TOTAL COUNTS ###
#     # prepare the result string with total counts
#     result = id OFS
#     for (catnum in categories) {
#         if (categories[catnum] in target_categories) {
#             if (NF == 1) { # handle case where there are no message tokens
#                 result = result "0" OFS
#             } else {
#                 result = result sum_hits[catnum] OFS # use the total count
#             }
#         }
#     }
#     print result # print the result
# }
#################### END OF MAIN BLOCK ####################