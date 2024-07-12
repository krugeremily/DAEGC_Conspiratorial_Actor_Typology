#!/usr/bin/gawk #specify interpreter

#THIS SCRIPT OUTPUT A COUNT OF HOW OFTEN A WORD OCCURS PER CATEGORY OVER THE ENTIRE CORPUS

#################### BEGIN BLOCK ####################
#we set the field separator,output record seperator & output field separator to tab; needed when printing an array
#we also set the maxchar and minchar; used later to track max and min length of entity
BEGIN {FS="\t";ORS="\t";OFS="\t";maxchar=0;minchar=100}
#################### END OF BEGIN BLOCK ####################

#################### MAIN BLOCK ####################
{

########## THIS PART IS ONLY DONE ON FIRST INPUT FILE AKA DICTIONARY ##########
        #NR==FNR ist an awk trick to make sure that we stay in the first file; here, our dictionary file
        #NR =total number of input records processed so far across all input files; FNR = number of input records processed so far in the current file
        #when you have multiple input files, NR continues to increment across all files, while FNR resets to 1 for each new file
        if(NR==FNR) 
    { 
        #if first field contains a wildcard at the end we do special substring matching later
        #checks if first field ends in asterisk
        if($1~/\*$/){
            #we remove the wildcard at the end of the liwc word with substr(ing)
            #get field three of dict
            pattern[tolower(substr($1,1,length($1)-1))][$3]
            
            if (length($1)>maxchar){ #update maxchar if length of word is greater than current maxchar
            maxchar=length($1)
            }
            
            if (length($1)<minchar){ #update minchar if length of word is less than current minchar
            minchar=length($1)
            }
            
            }
            
        else{
            pattern_exact[tolower($1)][$3] #getfield three of dict
        }
        
        categories[$3+0]=$2 #initialize categories array with field 3 of dict as key and field 2 of dict as value
        
        #for the first file, we just build our lookup table and forget about the stuff below
        next
        
        }
########## END OF CODE FOR FIRST INPUT FILE AKA DICTIONARY ##########


########## THIS PART IS DONE ON BOTH INPUT FILES ##########

##### the following currently commented out because we dont exclude anyting; #####
##### if block should be run, initialize exclude in BEGIN part or inline command #####

#check if first record of current file      
# if(FNR==1){
#     printf("category\tword\tfreq\n")
#     #if excludsion word are initialized in exclude, these terms are essentially "deleted" from the dictionary
#     split(exclude,excludearray,",")
#     for(e in excludearray){
#         delete pattern[excludearray[e]]
#         delete pattern_exact[excludearray[e]]
#     }
# }

#we loop over each field (one field = one token)
for(i=1;i<=NF;i++)
{
    
    #if there is an exact match, we count 1 for each category a word belongs to and go to the next increment in the loop
    if ($i in pattern_exact)
    {

        for(l in pattern_exact[$i])
        {
            sum_hits[l][$i]++
        }
    continue
    
    }
    
    #if there is no exact match, we reduce the string by one and try to match it to the wildcard words on each iteration, 
    #if there is a match we do the ususal counting and break this character-reducing loop, this means we find the longest matching string
    #* reduce the word in the first iteration by the amount that makes it the same length as the longest word our wildcard dictionary
    #* stop when you have reached the minimum length in the wildcard dictionary
    
    for(s=maxchar;s>=minchar;s--){
    if(substr($i,1,s) in pattern){
        
        for (l in pattern[substr($i,1,s)]){
            sum_hits[l][substr($i,1,s)"*"]++
#             show the actual matches? could be useful for debugging the lexicon
#             sum_hits[l][$i]++
        }
        break
        }
    }
}

}
#################### END OF MAIN BLOCK ####################


#################### END BLOCK ####################
END{

for(e in sum_hits)
{
for(w in sum_hits[e]){
 print e,w,sum_hits[e][w]
 printf "\n"
}
} 
}
#################### END OF END BLOCK ####################