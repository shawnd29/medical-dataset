# medical-dataset

This is a simulated dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features). Amongst all SNPs are 15 causal 
ones which means they and neighboring ones discriminate between case and 
controls while remainder are noise.

In the training are 4000 cases and 4000 controls. Your task is to predict 
the labels of 2000 test individuals.

In this progam, we used Chi-Suqre modelling to find the goodness of fit for the
15 casual features and based the predictions related to the dataset 
