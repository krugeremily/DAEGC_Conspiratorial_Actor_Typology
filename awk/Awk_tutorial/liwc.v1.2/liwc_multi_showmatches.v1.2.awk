#!/usr/bin/gawk

#we set the output record seperator, needed when printing an array
BEGIN {FS="\t";ORS="\t";OFS="\t";maxchar=0;minchar=100}
{
        #NR==FNR ist an awk trick to make sure that we stay in the first file here, our dictionary file
        if(NR==FNR)
    { 
        #if it contains a wildcard at the end we do special substring matching later
        if($1~/\*$/){
            #we remove the wildcard at the end of the liwc word with substr(ing)
            pattern[tolower(substr($1,1,length($1)-1))][$3]
            
            if (length($1)>maxchar){
            maxchar=length($1)
            }
            
            if (length($1)<minchar){
            minchar=length($1)
            }
            
            }
            
        else{
            pattern_exact[tolower($1)][$3]
        }
        
        categories[$3+0]=$2
        
        #for the first file, we just build our lookup table and forget about the stuff below
        next
        
        }

       
if(FNR==1){
    printf("category\tword\tfreq\n")
    split(exclude,excludearray,",")
    for(e in excludearray){
        delete pattern[excludearray[e]]
        delete pattern_exact[excludearray[e]]
    }
}

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

END{

for(e in sum_hits)
{
for(w in sum_hits[e]){
 print e,w,sum_hits[e][w]
 printf "\n"
}
} 
}