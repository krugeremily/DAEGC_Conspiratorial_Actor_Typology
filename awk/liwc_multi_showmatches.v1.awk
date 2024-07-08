#!/usr/bin/gawk

#as the first line, we print the categories that were chosen (as a header)
#update: no header, to not have problems with cat
#next update: again header, to avoid errors, cat with awk and NR>1
#BEGIN {FS="\t";split(cat,categories," "); for(e in categories) printf("\"%s\"\t",categories[e]); print "\"NF\"";}

# 1: "anger"
# 2: "anxiety"
# 3: "sadness"
# 4: "prosocial"
# 5: "social"
# 6: "posemo"


#we set the output record seperator, needed when printing an array
BEGIN {FS="\t";ORS="\t";OFS="\t"}
# BEGIN {FS="\t"}
{
        #NR==FNR ist an awk trick to make sure that we stay in the first file here, our dictionary file
        if(NR==FNR)
    { 
        #if it contains a wildcard at the end we do special substring matching later
        if($1~/\*$/){
        for(i=2;i<=NF;i++)
        {
            #we remove the wildcard at the end of the liwc word with substr(ing)
            pattern[tolower(substr($1,1,length($1)-1))][$i]
            }
        }
        else{
        for(i=2;i<=NF;i++){
                pattern_exact[tolower($1)][$i]
            }
        }
        
        #for the first file, we just build our lookup table and forget about the stuff below
        next
        
        }

# for(e in pattern_exact){
#     print e
#     for(i in pattern_exact[e]){
#         print i
#     }
# }
        
if(FNR==1){
    printf("category\tword\tfreq\n")
}

#we set the hit count to 0

# for(i=01;i<=06;i++){
#     sum_hits[i]=0
# }

# Instantiate arrays
# for(i in sum_hits){
#     for(e in pattern){
#         sum_hits[i][e]=0
#     }
#     for(e in pattern_exact){
#         sum_hits[i][e]=0
#     }
# }


# For debugging

# for(e in sum_hits)
# {
# print e
# }
# print NF
# printf "\n"
# No Buffering, not needed anymore
# system("")

# exit

# for(c in categories)
# {
#     sum_hits[categories[c]]=0
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
    
    #if there is no exact match, we reduce the string by one and try to match it to the wildcard words on each iteration, if there is a match we do the ususal counting and break this character-reducing loop, this means we find the longest matching string
    #to do: there would be ways to make this more efficient; e.g.
    #* reduce the word in the first iteration by the amount that makes it the same length as the longest word our wildcard dictionary, UPDATE: longest word in liwc is 24 (lebensabschnittsgefaehrt) characters
    #* stop when you have reached the minimum length in the wildcard dictionary, UPDATE: shortest word in prosocial wildcards is 1 characters ("x")
    #* sort words in the dictionary by length and start matching with the longest?
    
    for(s=24;s>=1;s--){
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

END{

for(e in sum_hits)
{
for(w in sum_hits[e]){
 print e,w,sum_hits[e][w]
 printf "\n"
}
} 
}
# print NF
# 
# }