#!/usr/bin/gawk

#These are the chosen categories
# 1: "anger"
# 2: "anxiety"
# 3: "sadness"
# 4: "prosocial"
# 5: "social"
# 6: "posemo"


#we set the output record seperator, needed when printing an array
BEGIN {FS=" ";ORS="\t"}
{
        #NR==FNR ist an awk trick to make sure that we stay in the first file here, our dictionary file
        if(NR==FNR)
    { 
        #if word in dict contains a wildcard at the end we do special substring matching later
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
        
if(FNR==1){
    printf("anger\tanxiety\tsadness\tprosocial\tsocial\tposemo\tnumber_tokens\n")
}

#we set the hit count to 0
for(i=01;i<=06;i++){
    sum_hits[i]=0
}

#we loop over each field (one field = one token)
for(i=1;i<=NF;i++)
{
    
    #if there is an exact match, we count 1 for each category a word belongs to and go to the next increment in the loop
    if ($i in pattern_exact)
    {

        for(l in pattern_exact[$i])
        {
            sum_hits[l]++
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
            sum_hits[l]++
        }
        break
        }
    }
}


#we print the results
for(e in sum_hits)
{
print sum_hits[e]
}
print NF
printf "\n"
#No Buffering, not needed anymore
#system("")
}