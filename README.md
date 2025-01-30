# Testing Deep Attentional Embedded Graph Clustering (DAEGC) to Create a Typology of Conspiratorial Actors

## Overall

This is the (final) code accompanying my Master's Thesis for the M. Sc. in Social Data Science at the University of Copenhagen. The thesis was graded 10 (B).

In short: 
In the thesis (or rather, the code), I
- utlizes an anonymized dataset of Telegram messages from January 2021 from groups and channels that discuss one of these four conspiracy theories: QAnon, Pizzagate, Reichsbürger, Chemtrails
- extract 30 linguistic features from those messages
- build the author network by establishing edges between authors who wrote a message to the same group
- train a Graph Attentional Autoencoder and a Deep Embedded Attentional Graph Clustering Model to group the authors into 4 clusters
- assess the external and internal validity of those clusters 

## Thesis Abstract

This thesis investigates the potential of Attributed Graph Clustering, specifically Deep Attentional Embedded Graph Clustering (DAEGC) in unsupervised settings, by aiming to create a typology of conspiratorial actors on Telegram. Understanding the types and dynamics of conspiratorial actors is crucial in combatting the spread of harmful narratives. The DAEGC model is particularly promis-ing, as it incorporates both network structure and node features (here: linguistic features) in its clustering process, which has been shown to outperform methods that only utilize one of the two elements. In this thesis, the model demonstrates strong clustering performance, that, however, pri-marily relies on network structure rather than linguistic features. This is due to a lack of variability in the selected linguistic features and an overly broad edge definition, who in combination result in a highly connected network. This hinders the model’s ability to differentiate between distinct types of conspiratorial actors based on the linguistic features. The findings suggest that DAEGC, though effective in supervised settings, faces significant challenges in unsupervised contexts. This thesis highlights the need for more careful feature engineering and edge definition, as well as additional performance metrics in the training process of the DAEGC in unsupervised settings. Future work can build on the insights and limitations of this thesis by utilizing an interpretable clustering algorithm and carefully selecting and analyzing linguistic features and edges to create a typology of conspiratorial actors.
 
