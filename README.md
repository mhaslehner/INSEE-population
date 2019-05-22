
## Analyzing French socioeconomic data (income and education) using INSEE data


In this project, I analyze a dataset of socioeconomic indicators such as education and income
in France in the year 2013/2014 and perform a principal component analysis (PCA), and clustering, 
in order to retrieve information on the data.  

#### I. The dataset

I use two datasets taken from the French INSEE (Institute of statistics), 
the first one on income and poverty, 'Revenus et pauvreté des ménages en 2013' (https://www.insee.fr/fr/statistiques/2388572), 
and the second one on education, 'Population des 16 ans ou plus selon le niveau de diplôme, le sexe et l'âge de 1968 à 2014'
(https://www.insee.fr/fr/statistiques/1893149). Since for the second file, the year 2013 was not available, I chose the year 2014.
I use the data ordered by French département (corresponding sheet in the excel files).

The first dataset consists of 28 columns, containing information such as percentage of poverty among the total population, 
median living standard, percentage of work revenues, ratio of the 9th to the 1st percentile in terms of living standard, 
etc. From the second dataset related to education, I only retrieved columns corresponding to 'University degree' vs 'no degree'.



#### II. A first glance at the data - looking for distributions and correlations

For the analysis, I concatenate the columns from the 'education file' with those from the 'income file'. 
The dataset under consideration results in a matrix containing 28 columns and 96 rows (départements from  
metropolitan France).

Since the values in the different columns have very different magnitudes, 
it makes sense to normalize the data for an easier comparison. 
I calculate (X-E[X]/sigma(X)), where E and sigma denote the mean and the standard deviation (over the 96 départements), 
respectively.  





#### III. PCA and Clustering


explained variance ratio [0.48658931 0.25103022 0.12192411] 