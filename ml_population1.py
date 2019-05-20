#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8


from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
import os                         # For issuing commands to the OS.
import numpy as np
import re
import unicodedata
import time
import random
from matplotlib.font_manager import FontProperties
import pandas as pd
import csv
from scipy import stats
import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import datetime
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import brier_score_loss
from scipy.stats import bernoulli
#from mpl_toolkits.basemap import Basemap
import xlrd
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
#import pygal_maps_fr?
import pygal.maps.fr
from sklearn.decomposition import PCA
from numpy.testing import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets





# assert_array_equal(arr1,arr2)
##########################################################################################


#  1. Read in (excel) data sets


##########################################################################################

income_cols = pd.read_excel('filo-revenu-pauvrete-menage-2013.xls',sheet_name='DEP',sep=';', header = 3)    #,header=None
income = pd.read_excel('filo-revenu-pauvrete-menage-2013.xls',sheet_name='DEP',sep=';', header = 5)    #,header=None
#
print('---------------- 1.) read in income file -----------------------')
print('')
print('revenus et pauvreté des ménages en 2013')

income_cols.info()
col_names = income_cols.iloc[0,2:] # need to start with same column indices as for income data
dept_pov = [income['CODGEO'].iloc[i] for i in range(96)]  #
#print(col_names)

#print(income.describe())

print('--------------- 2.) read in diploma file ------------------------')
print('')
print(' diplomes ')

#diploma_cols = pd.read_excel('pop-16ans-dipl6814.xls', sheet_name='DEP_2014',sep=';', header = 13)
diplom = pd.read_excel('pop-16ans-dipl6814.xls', sheet_name='DEP_2014',sep=';', header = 14)

#dipl = pd.Dataframe()
#print(diplom['dpx_recAs1age2_rec1rpop2014'])

diplom.info()
dept_dipl = [diplom['DR16'].iloc[i] for i in range(96)]

print('---------------- 3.) add diploma women and men for the four eductadion categories A,B,C,D ---------------------')

# for the diploma, there are just two age intervals; < 25 yrs or > 25 yrs..
# > 25 yrs 'dpx_recAs1age2_rec2rpop2014' vs _rec1rpop2014 for young people <25y
age = '2' # for diploma

# A ..no diploma
dipl_A_men_and_women = [(diplom['dpx_recAs1age2_rec'+str(age)+'rpop2014'].iloc[i]+diplom['dpx_recAs2age2_rec'+str(age)+'rpop2014'].iloc[i]) for i in range(96)]

dipl_A = zip(dept_dipl,dipl_A_men_and_women)
diploma_A = dict(dipl_A)
print(diploma_A)
# B .. CAP or BEP
dipl_B_men_and_women = [(diplom['dpx_recBs1age2_rec'+str(age)+'rpop2014'].iloc[i]+diplom['dpx_recBs2age2_rec'+str(age)+'rpop2014'].iloc[i]) for i in range(96)]
dipl_B = zip(dept_dipl,dipl_B_men_and_women)
diploma_B = dict(dipl_B)
#print(diploma_B)
# C .. bac/highschool
dipl_C_men_and_women = [(diplom['dpx_recCs1age2_rec'+str(age)+'rpop2014'].iloc[i]+diplom['dpx_recCs2age2_rec'+str(age)+'rpop2014'].iloc[i]) for i in range(96)]
dipl_C = zip(dept_dipl,dipl_C_men_and_women)
diploma_C = dict(dipl_C)
#print(diploma_C)
# D .. University
dipl_D_men_and_women = [(diplom['dpx_recDs1age2_rec1rpop2014'].iloc[i]+diplom['dpx_recDs2age2_rec1rpop2014'].iloc[i]) for i in range(96)]
dipl_D = zip(dept_dipl,dipl_D_men_and_women)
diploma_D = dict(dipl_D)
#print(diploma_D)
# for the income, there are several age intervals:
# TP60AGE113 <30ans age 1 .. will be compared to diploma of young people <25yrs
# TP60AGE213 30-39ans age 2
# TP60AGE313 40-49ans age 3

#age_income = '1' # < 30yrs
#age_income = '2' # < 30-39yrs
age_income = '3' # < 40-49yrs
#age_income = '4' # < 50-59yrs
if age_income == '3':
   ag = '40-49'


# the columns of diploma and income are not ordered in the same way following the departement, i.e the dept orderings are
# different for both files.. we need to adapt the column entries so that for both files, each row corresponds to the same
#  department. In order to do that, we choose a reference ordering (say, as in the file income), and adapt the ordering
# of the other file (diploma) according to this reference
# create a dictionary starting with tuples of (diploma, income) for each row
#array_dipl_income_groupA = [(diploma_A[k],income_l30[k]) for k in income_l30.keys()]
income30 = [income['TP60AGE'+str(age_income)+'13'].iloc[i] for i in range(96)] # poor people under 30 < yrs
income_less30 = zip(dept_pov,income30)
income_l30 = dict(income_less30)
#print('dictionary made with entries of income and department',income_l30)
#
#
#
#
#-----------------------------------------------------------------------------------------
#
# 1. Dimensionality reduction: PCA
#
#-----------------------------------------------------------------------------------------

print('---------------- 4.) perform a PCA analysis on the poverty/income data ---------------------')

# astype is necessary to avoid error, since np.asarray does not work fully on a pandas object
# https://stackoverflow.com/questions/47208473/attributeerror-numpy-float64-object-has-no-attribute-log10

#
income_all =  [income.iloc[i,2:] for i in range(96)]                # we start with the 3rd column
income_array = np.asarray(income_all).astype(np.float64)            # convert income pandas dataframe into a numpy array


#-----------------------------------------------------------------------------------------------------------------------
#
# 1.a. Concatenate the income file and (some columns of the) education file (for now only uneducated and graduated
# people > 25yrs)
#
#-----------------------------------------------------------------------------------------------------------------------

# append education (for now only 'no education' and 'University degree', A and D as defined above)
# the column (departments) entries of the education matrix need to be in the right order (same as for income)

# take any of the income columns and create a dictionary with the dept column.
income_l30 = [income['TP60AGE'+str(age_income)+'13'].iloc[i] for i in range(96)] # poor people under 30 < yrs
income_key = zip(dept_pov,income_l30)
income_keys = dict(income_key)

# adapt chronology of (changed) diploma keys to (unchanged) income keys;
# take only diploma_A (little education) and diploma_D (University degree) for men and women aged > 25years.
#array_dipl_income_groupA = [(diploma_A[k],income_l30[k]) for k in income_l30.keys()]
new_diploma = [(diploma_A[k],diploma_D[k]) for k in income_keys.keys()]
print(new_diploma)
# would need to check the keys are right --> print concerned keys and compare with files ...
# make a np array out of this and add the diploma entries to the income array
new_diploma_older25y = np.asarray(new_diploma)

print('income array',new_diploma_older25y.shape,type(new_diploma_older25y))

# then ..concatenate income_array
income_all_more25 = np.concatenate((new_diploma_older25y,income_array),axis=1)
print(income_array.shape)
print(income_all_more25.shape)
#exit()

# create a function that normalizes the values of a matrix, column-wise, by dividing by the std over all departments
# (the mean and the standard deviation of all column elements are computed)
#
def normalized_array(X_test): # X is a (n,m) .. (samples/depts,features)-dim matrix
    centered = X_test - X_test.mean(axis = 0)
    #print('mean(X)',X_test.mean(axis = 0))
    st = np.std(centered,axis=0)
    #print('standard deviation',st)
    normalized = centered/st
    #print(normalized)
    #exit()
    return normalized


#-----------------------------------------------------------------------------------------------------------------------
#
# 1.b. Perform the PCA with the concatenated file (income_all_more25) using normalization of the data values
#
#-----------------------------------------------------------------------------------------------------------------------


#all = income_array.shape[1] ?
all = 3 # number of principal components we select
pca = PCA(n_components = all)
Normal_income = normalized_array(income_all_more25) # N = (96,26)
X_reduced = pca.fit_transform(Normal_income)
S = X_reduced[:,0] # the projection onto the first principal component of the normalized data Normal_income



print('shape of outcome pca',X_reduced.shape)
for l in range(3):
    print('pca vector',l,pca.components_.T[:,l],' proj(income)_on_pca',X_reduced[:,l])
#exit()
#print('projection of income on the vector',)
# how to assign names of variables in a loop?
# build a 'matrix' made of principal components
#c1 = pca_income.components_.T[:,0]  # .T gives a (26,)= (26,1) vector
#--------------------------------------------
var_values = pca.explained_variance_ratio_
print('explained variance ratio',var_values)
singular_values = pca.singular_values_
print('singular values',singular_values)
#--------------------------------------------
# The vector
# S = c1*N1 + c2*N2 +... + cn*Nn, where c1,c2,..,cn are the coordinates of a single pca vector.
# and Ni are the normalized columns of the initial data
# was calculated by pca_fit..


#-----------------------------------------------------------------------------------------------------------------------
#
# 1b. Plot PCA1 and PCA2, as well as a measure for socioeconomic strength S on a geographic map of France
#
#-----------------------------------------------------------------------------------------------------------------------

#income_array[]
size_ = 40
fig = plt.figure() # figsize=(10,10)
#plt.title('cluster of poor ('+str(ag)+'y) vs uneducated (> 25y) people')
X1 = X_reduced[:,0]
Y1 = X_reduced[:,1]
plt.scatter(X1,Y1, s=size_, c='blue')        #,label='no education'
#plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],linestyle='--')
#plt.axis('tight')
#plt.legend()
plt.ylabel('pca3 (from normalized data)')
plt.xlabel('pca2 ( from normalized data)')
fig.savefig('PCA1_and_2_for_extended_file.png')
print('plotted pca final')
#exit()

for i in range(len(S)):
    print(S[i])
print('max',S.max())
print('min',S.min())
#exit()
# create a dictionary made from tuples (departement,S)
#
zip_strength = zip(dept_pov,S)
socioec_strength = dict(zip_strength)
print(socioec_strength)
#array_dipl_income_groupA = [(diploma_A[k],income_l30[k]) for k in income_l30.keys()]
#dict.keys()
#dict.values()
#dict.items()

# Plot this index according to the département.
#
print('plot')
fr_chart = pygal.maps.fr.Departments()
fr_chart.title = 'Socioeconomic strength in France (2013)'
#fr_chart.add('Métropole', ['69', '92', '13'])
#fr_chart.add('Corse', ['2A', '2B'])
#fr_chart.add('DOM COM', ['971', '972', '973', '974'])
#fr_chart.render_to_file('pygal_test.svg')
#fr_chart.render_to_png('pygal_test.png')
# assign a number to each departement
#income_l30 = income['TP60AGE113']
#fr_chart = pygal.maps.fr.Departments(human_readable=True)
#fr_chart.title = 'Ratio of poor people younger than 30 years'
fr_chart.add('PCA1 value',socioec_strength)
             #{
    #dept, s_value for (dept, s_value) in socioec_strength.items()
  # '01': income_l30[0], '02'
   #})
#i = 0
print('plot')
#for (dept,s_value) in socioec_strength.items():
    #i = i+1
    #if i<20:
#    fr_chart.add(dept, s_value)
    #print(dept,s_value)
fr_chart.render_to_png('pygal_test_Socioeconomic_Strength_extendedfile_PCA1.png')
#exit()
#-----------------------------------------------------------------------------------------------------------------------
#
# 1c. Plot 3D space of principal components
#
#-----------------------------------------------------------------------------------------------------------------------

# import some data to play with
#iris = datasets.load_iris()
# data
#X = iris.data[:, :2]  # we only take the first two features.
#y = iris.target
#x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#plt.figure(2, figsize=(8, 6))
#plt.figure()
#plt.clf()
# Plot the training points
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
#            edgecolor='k')
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
#plt.xticks(())
#plt.yticks(()

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions

elev = [-150,30]
azim = [-110,50]
for i in range(2):
   fig = plt.figure(1, figsize=(8, 6))

   ax = Axes3D(fig, elev=elev[i], azim=azim[i]) # Azimuthal viewing angle, defaults to -60, Elevation viewing angle, defaults to 30.

   ax.plot(X_reduced[:, 0], X_reduced[:, 2], 'ko',markersize=1.5, zdir='y',zs=-10)
   ax.plot(X_reduced[:, 1], X_reduced[:, 2], 'ko',markersize=1.5, zdir='x',zs=-10)
   ax.plot(X_reduced[:, 0], X_reduced[:, 1], 'ko',markersize=1.5, zdir='z',zs=-10)

   #X_reduced = PCA(n_components=3).fit_transform(iris.data)
   #colors = np.random.rand(len(X_reduced[:, 0]))

   ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],cmap=plt.cm.Set1, s=50)


   #ax.plot(x, z, 'r+', zdir='y', zs=1.5)
   #ax.plot(y, z, 'g+', zdir='x', zs=-0.5)
   #ax.plot(x, y, 'k+', zdir='z', zs=-1.5)

   ax.set_xlim([-10., 10.])
   ax.set_ylim([-10., 10.])
   ax.set_zlim([-10., 10.])

   #ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 0], cmap=plt.cm.Set1, edgecolor='k', s=40)
   #ax.scatter(X_reduced[:, 1], X_reduced[:, 1], X_reduced[:, 0], cmap=plt.cm.Set1, edgecolor='k', s=40)
   #ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 0], cmap=plt.cm.Set1, edgecolor='k', s=40)



   #z_max = max(X_reduced[:, 2])
   #ax.set_zlim(-1, )
   ax.set_title("First three PCA directions")
   ax.set_xlabel("1st principal component")
   ax.w_xaxis.set_ticklabels([])
   ax.set_ylabel("2nd principal component")
   ax.w_yaxis.set_ticklabels([])
   ax.set_zlabel("3rd principal component")
   ax.w_zaxis.set_ticklabels([])
   fig.savefig('3Dplot_PCA123_for_extended_file'+str(i)+'.png')
   print('plotted pca final all')




exit()
#----------------------------------------------------------------------

for i in range(len(col_names)):
    print(i,col_names[i])
#exit()

a = 22
b = 4

#income_array[]
size_ = 40
fig = plt.figure() # figsize=(10,10)
#plt.title('cluster of poor ('+str(ag)+'y) vs uneducated (> 25y) people')
X1 = X[:,a]
Y1 = X[:,b]
plt.scatter(X1,Y1, s=size_, c='blue')        #,label='no education'
#plt.scatter(X,Z , s=size_, c='royalblue')
#plt.scatter(Y,Z , s=size_, c='deepskyblue')

#plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],linestyle='--')
#plt.axis('tight')
#plt.legend()
plt.ylabel(''+str(col_names[b])+' (normalized)')
plt.xlabel(''+str(col_names[a])+' (normalized)')
fig.savefig('finding_pca_colums_'+str(a)+''+str(b)+'.png')
print('plotted pca pre')

exit()


# #############################################################################
# Plot the figures
#def plot_figs(fig_num, elev, azim):
#    fig = plt.figure(fig_num, figsize=(4, 3))
#    plt.clf()
#    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)

#    ax.scatter(income_array[:,c1], income_array[:,c2], income_array[:,c3], c=density[::10], marker='+', alpha=.4)
#    Y = np.c_[income_array[:,c1], income_array[:,c2], income_array[:,c3]]

    # Using SciPy's SVD, this would be:
    # _, pca_score, V = scipy.linalg.svd(Y, full_matrices=False)

#    pca = PCA(n_components=3)
#    X_income = pca.fit_transform(Y)

#    pca = PCA(n_components=3)
#    pca.fit(Y)
#    pca_score = pca.explained_variance_ratio_
#    V = pca.components_

#    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V.T
#    x_pca_plane = np.r_[x_pca_axis[:2], - x_pca_axis[1::-1]]
#    y_pca_plane = np.r_[y_pca_axis[:2], - y_pca_axis[1::-1]]
#    z_pca_plane = np.r_[z_pca_axis[:2], - z_pca_axis[1::-1]]
#    x_pca_plane.shape = (2, 2)
#    y_pca_plane.shape = (2, 2)
#    z_pca_plane.shape = (2, 2)
#    ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
#    ax.w_xaxis.set_ticklabels([])
#    ax.w_yaxis.set_ticklabels([])
#    ax.w_zaxis.set_ticklabels([])


#elev = -40
#azim = -80
#plot_figs(1, elev, azim)

#elev = 30
#azim = 20
#plot_figs(2, elev, azim)

#plt.show()



exit()









# the columns of diploma and income are not ordered in the same way following the departement, i.e the dept orderings are
# different for both files.. we need to adapt the column entries so that for both files, each row corresponds to the same
#  department. In order to do that, we choose a reference ordering (say, as in the file income), and adapt the ordering
# of the other file (diploma) according to this reference
# create a dictionary starting with tuples of (diploma, income) for each row
array_dipl_income_groupA = [(diploma_A[k],income_l30[k]) for k in income_l30.keys()]
array_dipl_income_groupB = [(diploma_B[k],income_l30[k]) for k in income_l30.keys()]
array_dipl_income_groupC = [(diploma_C[k],income_l30[k]) for k in income_l30.keys()]
array_dipl_income_groupD = [(diploma_D[k],income_l30[k]) for k in income_l30.keys()]

#print('',income_l30.keys())
print('A',array_dipl_income_groupA)
print('B',array_dipl_income_groupB)
print('C',array_dipl_income_groupC)
print('D',array_dipl_income_groupD)

#-----------------------------
# Test if this is working: creating two arrays of values that correspond to the same keys.
# the values corresponding to '2A','2B' (Corsica) of the file diploma should correspond to those values (at different line of the file)
# of the file income
#
# i = 0
#for key in income_l30.keys():
#    print('key of income',key,'diploma',diploma_A[key],'income',income_l30[key],'new tuple',diploma_[i])
#    i = i+1
# exit()
#-----------------------------
# then make np arrays out of tuple
# convert list of tuples to array

array_A = np.asarray(array_dipl_income_groupA)
array_B = np.asarray(array_dipl_income_groupB)
array_C = np.asarray(array_dipl_income_groupC)
array_D = np.asarray(array_dipl_income_groupD)

#print('New Array',array_A)
#print(array_A.shape)

array_diplomaA = array_A[:,0]
array_povertyA = array_A[:,1]
array_diplomaB = array_B[:,0]
array_povertyB = array_B[:,1]
array_diplomaC = array_C[:,0]
array_povertyC = array_C[:,1]
array_diplomaD = array_D[:,0]
array_povertyD = array_D[:,1]


print('array of diploma',array_diplomaA)
print('array of poverty',array_povertyA)

#-----------------
# Correlation between non-education and poverty? scatter plot
# don't see any correlation. Do we need to look at higher ages?
plt.figure()
plt.plot(array_povertyA, array_diplomaA,'b*',markersize = 6, label =('No education'))
#plt.plot(array_povertyB, array_diplomaB,'g*',markersize = 6, label =('CAP or BEP'))
#plt.plot(array_povertyC, array_diplomaC,'r*',markersize = 6, label =('High school (bac)'))
plt.plot(array_povertyD, array_diplomaD,'y*',markersize = 6, label =('University'))
#plt.plot(inc_less30,Regr_line_maxt_cos[:c],'b-',linewidth = 1.0,label=('k = %4.2f +/- %4.2f' %(slope_maxt_cos,std_maxt_cos)))
plt.xlabel('Rate of poverty ('+str(ag)+'y) [in % for each dept]')
plt.ylabel('Number of individuals/dept')
plt.title('poverty rate ('+str(ag)+'y) vs nb of (un-)educated people > 25yrs')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$') # text in the graphic
#plt.axis([10, 40, 0, 0.03])
plt.legend() #fontsize=8
plt.tight_layout()
plt.grid(True)
plt.savefig('poverty_vs_nodiploma_young_age'+str(age_income)+'.png')



#------ Rate of poverty among educated vs non educated people?; histograms

# the histogram of the data
x1 = array_diplomaA # no educ
x2 = array_diplomaD # university
nb_of_bins = 30
maxi = max(max(x1),max(x2))
mini = min(min(x1),min(x2))

bin = (maxi - mini)/float(nb_of_bins) # bin length
bin_width = bin * 1.001  # if last point is equal to max, won't get error
x_grid = np.linspace(mini, maxi, nb_of_bins)  # creates a 1-dimensional array of 100 values between minim and maxim
x_values = np.fromfunction(lambda i: mini + i * bin_width, (nb_of_bins,)) # for plotting
kernel = np.zeros(nb_of_bins)

hist1 = stats.gaussian_kde(x1, bw_method=None)
hist2 = stats.gaussian_kde(x2, bw_method=None)
print('diplA',x1)
print('diplD',x2)
kernel1 = hist1.evaluate(x_grid)
kernel2 = hist2.evaluate(x_grid)
# plt.switch_backend('agg') # if error message for plot comes..
plt.figure()
n, bins, patches = plt.hist(x1, nb_of_bins, density=True, facecolor='g') # no edu
n, bins, patches = plt.hist(x2, nb_of_bins, density=True, facecolor='r') # univ

plt.plot(x_values[:],kernel1[:],'g-',linewidth=1.8,label ='no education')
plt.plot(x_values[:],kernel2[:],'r-',linewidth=1.8,label='university')

plt.xlabel('Number of individuals')
plt.ylabel('Probability')
plt.title('Histogram of (un-)educated people (>25y) per French departement')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$') # text in the graphic
#plt.axis([10, 40, 0, 0.03])
plt.ylim(0,6.0e-5)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('histo_edu_older25_bins.png')


# create clusters
X = array_A # X = (n_samples, n_features) = (samples of diploma among departments, samples of poverty ratio among dpts)
Y = array_D
hc = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')
size_ = 30
y_hc = hc.fit_predict(X) #
y_hc1 = hc.fit_predict(Y) #
print(hc.labels_)

fig = plt.figure() # figsize=(10,10)
plt.title('cluster of poor ('+str(ag)+'y) vs uneducated (> 25y) people')

#plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=size_, c='blue',label='no education')
#plt.scatter(X[y_hc==1,0], X[y_hc == 1,1], s=size_, c='royalblue')
#plt.scatter(X[y_hc ==2,0], X[y_hc == 2,1], s=size_, c='deepskyblue')
#plt.scatter(X[y_hc ==3,0], X[y_hc == 3,1], s=size_, c='aqua')
#plt.scatter(Y[y_hc1 == 0,0], Y[y_hc1 == 0,1], s=size_, c='r',label ='Univ deg')
#plt.scatter(Y[y_hc1==1,0], Y[y_hc1 == 1,1], s=size_, c='magenta')
#plt.scatter(Y[y_hc1 ==2,0], Y[y_hc1 == 2,1], s=size_, c='hotpink')
#plt.scatter(Y[y_hc1 ==3,0], Y[y_hc1 == 3,1], s=size_, c='pink')


plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=size_, c='blue',label='no education')
plt.scatter(X[y_hc==1,0], X[y_hc == 1,1], s=size_, c='royalblue')
plt.scatter(X[y_hc ==2,0], X[y_hc == 2,1], s=size_, c='deepskyblue')

#plt.scatter(Y[y_hc1 == 0,0], Y[y_hc1 == 0,1], s=size_, c='r')
#plt.scatter(Y[y_hc1==1,0], Y[y_hc1 == 1,1], s=size_, c='deeppink',label ='Univ deg')
#plt.scatter(Y[y_hc1 ==2,0], Y[y_hc1 == 2,1], s=size_, c='lavenderblush')

# what would it be for 5 clusters?
#plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],linestyle='--')
#plt.axis('tight')
plt.legend()
plt.ylabel('Percentage of poor people ('+str(ag)+'y)')
plt.xlabel('Nb of uneducated individuals (> 25y) (per department)')
fig.savefig('first_cluster_pov_unedu_age'+str(age_income)+'.png')
print('plotted cluster')

exit()




#----------------------------------------------------------------------------
#
# plot histograms of poor population (over French departements)
#
#------------------------------------------------------------------------------


# the histogram of the data
x = income_l30
nb_of_bins = 30
maxi = 36
mini = 13
bin = (maxi - mini)/float(nb_of_bins) # bin length
bin_width = bin * 1.001  # if last point is equal to max, won't get error
x_grid = np.linspace(mini, maxi, nb_of_bins)  # creates a 1-dimensional array of 100 values between minim and maxim
x_values = np.fromfunction(lambda i: mini + i * bin_width, (nb_of_bins,)) # for plotting
kernel = np.zeros(nb_of_bins)
hist = stats.gaussian_kde(x, bw_method=None)
kernel = hist.evaluate(x_grid)
# plt.switch_backend('agg') # if error message for plot comes..
plt.figure()
n, bins, patches = plt.hist(x, nb_of_bins, density=True, facecolor='g')
plt.plot(x_values[:],kernel[:],'g-',linewidth=1.8)
plt.xlabel('Rate of poverty (< 30y) [in %]')
plt.ylabel('Probability')
plt.title('Histogram of poverty rate (< 30y) among French departements')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$') # text in the graphic
#plt.axis([10, 40, 0, 0.03])
plt.grid(True)
plt.savefig('histo_population30_30bins.png')


exit()


#----------------------------------------------------------------------------
#
# pplot France with départements using pygal
#
#------------------------------------------------------------------------------


# int1 13-17, int2 17-22, int3 22-26, int4 26-30, int5 30-34
# for j in int:
#    for i in len(income_l30):
#       if income_l30[i] < int[i+1] and income_l30[i] > int[i]:
#            retrieve corresponding department i (dict[i][:])
#            dict.keys()[i]
# >>> all_dsc(42).keys()
# ['a', 'b']


fr_chart = pygal.maps.fr.Departments()
fr_chart.title = 'Some departments'
fr_chart.add('Métropole', ['69', '92', '13'])
fr_chart.add('Corse', ['2A', '2B'])
fr_chart.add('DOM COM', ['971', '972', '973', '974'])
fr_chart.render_to_file('pygal_test.svg')
fr_chart.render_to_png('pygal_test.png')
# assign a number to each departement
income_l30 = income['TP60AGE113']
fr_chart = pygal.maps.fr.Departments(human_readable=True)
fr_chart.title = 'Ratio of poor people younger than 30 years'
fr_chart.add('In 2015', {
  '01': income_l30[0], '02': income_l30[1], '03': income_l30[2], '04': income_l30[3], '05': income_l30[4], '06': income_l30[5],
    '07': income_l30[6], '08': income_l30[7], '09': income_l30[8], '10': income_l30[9], '11': income_l30[10], '12': income_l30[11],
    '13': income_l30[12], '14': income_l30[13], '15': income_l30[14], '16': income_l30[15], '17': income_l30[16], '18': income_l30[17],
    '19': income_l30[18], '2A': income_l30[19], '2B': income_l30[20], '21': income_l30[21], '22': income_l30[22], '23': income_l30[23],
    '24': income_l30[24], '25': income_l30[25],'26': income_l30[26], '27': income_l30[27], '28': income_l30[28], '29': income_l30[29],
'30': income_l30[30], '31': income_l30[31],'32': income_l30[32], '33': income_l30[33], '34': income_l30[34], '35': income_l30[35],
'36': income_l30[36], '37': income_l30[37],'38': income_l30[38], '39': income_l30[39], '40': income_l30[40], '41': income_l30[41],
'42': income_l30[42], '43': income_l30[43],'44': income_l30[44], '45': income_l30[45], '46': income_l30[46], '47': income_l30[47],
'48': income_l30[48], '49': income_l30[49],'50': income_l30[50], '51': income_l30[51], '52': income_l30[52], '53': income_l30[53],
'54': income_l30[54], '55': income_l30[55],'56': income_l30[56], '57': income_l30[57], '58': income_l30[58], '59': income_l30[59],
'60': income_l30[60], '61': income_l30[61],'62': income_l30[62], '63': income_l30[63], '64': income_l30[64], '65': income_l30[65],
'66': income_l30[66], '67': income_l30[67],'68': income_l30[68], '69': income_l30[69], '70': income_l30[70], '71': income_l30[71],
'72': income_l30[72], '73': income_l30[73],'74': income_l30[74], '75': income_l30[75], '76': income_l30[76], '77': income_l30[77],
'78': income_l30[78], '79': income_l30[79],'80': income_l30[80], '81': income_l30[81], '82': income_l30[82], '83': income_l30[83],
'84': income_l30[84], '85': income_l30[85],'86': income_l30[86], '87': income_l30[87], '88': income_l30[88], '89': income_l30[89],
'90': income_l30[90], '91': income_l30[91],'92': income_l30[92], '93': income_l30[93], '94': income_l30[94], '95': income_l30[95]
})
fr_chart.render_to_png('pygal_test_income30')

# '971': 404635, '972': 392291, '973': 237549, '974': 828581, '976': 212645

#worldmap_chart.title = 'Minimum deaths by capital punishement (source: Amnesty International)'


print('finished plotting')


#xy_chart = pygal.XY()
#xy_chart.add('test', [(1,2), (2,2)])
##xy_chart.render('test.svg')
#xy_chart.render_to_file('test.svg')

#fr_chart.render_to_png('chart.png') # Write the chart in the specified file
# or specify a department
#fr_chart = pygal.maps.fr.Departments(human_readable=True)
#fr_chart.title = 'Population by department's

exit()
#-----------------------------------------------------------------------------
#
# Hierarchical clustering
#
#-----------------------------------------------------------------------------


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(inc_less30)
print(hc.labels_)
fig = plt.figure(figsize=(10,10))
plt.title('cluster poor people under 30')
print('plot hierarchical clustering of under 30yold poor population')
#plt.scatter(points[y_hc == 0,0], points[y_hc == 0,1], s=100, c='red')
#plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
#plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
#plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')
#plt.scatter(inc_less30[y_hc == 0,0], inc_less30[y_hc == 0,1], s=100, c='red')
#plt.scatter(inc_less30[y_hc==1,0], inc_less30[y_hc == 1,1], s=100, c='black')
#plt.scatter(inc_less30[y_hc ==2,0], inc_less30[y_hc == 2,1], s=100, c='blue')
#plt.scatter(inc_less30[y_hc ==3,0], inc_less30[y_hc == 3,1], s=100, c='cyan')
# what would it be for 5 clusters?
#plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],linestyle='--')
#plt.axis('tight')
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
fig.savefig('first_cluster2.png')
print('plotted cluster')
#fig = plt.figure(figsize=(10,10))




exit()


#pd.set_option('display.max_columns', None)


cols = profession.columns.tolist()
#cols = data_all2.columns
print(cols[:])
profession.head()
profession.shape
profession.info()


exit()

#---------------------------------- Clustering -------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
# followed tutorial by
# https://towardsdatascience.com/an-introduction-to-clustering-algorithms-in-python-123438574097

# 0. import KMeans
# from sklearn.cluster import KMeans

# Clustering (k-mean)
# 1. create kmeans object
# kmeans = KMeans(n_clusters=4)
# 2. fit kmeans object to data
#kmeans.fit(points)
# 3. print location of clusters learned by kmeans object
# print(kmeans.cluster_centers_)
# 4. save new clusters for chart
# y_km = kmeans.fit_predict(points)

# 5. Plot the 4 clusters
#plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='red')
#plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='black')
#plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='blue')
#plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=100, c='cyan')
#-----------------------------------------------------------------------------------------------------------------------

# Hierarchical clustering
# create clusters
# hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
# y_hc = hc.fit_predict(points)

#plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='red')
#plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
#plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
#plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')




#-----------------------------------------------------------------------------------------------------------------------

exit()
#print('',income.head)
#print(.inc)



#print('shape',income.shape[0])



exit()

print(income.iloc[0, :])
for i in range(income.shape[1]):
    print('', i, income.iloc[0, i])

def pov(file,col_ind,ind1 = None,ind2 = None):
   # define array either as integer (1st column) or as float 
   #print 'column index',col_ind
   Arr = file.values[1:,col_ind] 
   #print 'type Column',type(Arr)
   if col_ind == 2:
      Arr.astype(int)
   #elif col_ind == 1:
   #   Arr = np.array(Arr, dtype=string)
   else:         
      Arr = np.array(Arr, dtype=float)
   #print 'column',Arr
   return Arr



# how to populate a dictionary using a loop??
def foo(s1):
    return '%s' % s1

# will be the namelist of strings for the different column names of the poverty file
namelist = [foo(income.iloc[0, i].encode('ascii', 'ignore')) for i in range(28)] # create a namelist of string names that
                                    # will later serve as columns for the pandas array object...

print('namelist of column names',namelist,namelist[0])

#exit()
#print pov(poverty,0)
print('column 0', income.values[1:, 0].astype(int))
print('column 2', income.values[1:, 2])


dic = dict(zip(namelist, (pov(income, i) for i in range(2, 6))))
exit()

#print 'dict',dic
#exit()
#print colum_names = np.array([])
print('')
print('pauvreté ménages')
print('')
#df = pd.DataFrame(dic,columns=['CODGEO','NBMENFISC13','NBPERSMENFISC13'])
df = pd.DataFrame(dic,columns=['CODGEO'])
print(df)

exit()
#- old
#dic = {'A': pov(poverty,2),'B': pov(poverty,3), 'C':pov(poverty,4),'D':pov(poverty,5)}
#dict_v = {key:value for (key,value) in dic.items()}
#print 'dict_v',dict_v
#df = pd.DataFrame(dict_v,columns=['A','B'])
#print 'result',df # prints only chosen columns 'A' and 'B' among all those defined in dic.
#exit()
#------------------------------------------------------------------------------------------



# select columns to keep -----------------------------------------
#test_colmns = np.array([i for i in range(9)]) # keep all
#votes = pd.read_csv('Referendum.csv', sep=';',usecols = test_colmns, header = None)    #,header=None

#print 'referendum',votes.shape
#print 'values',votes.values[0,7]




# 1. Data analysis
# 2. Model training
# 3. Model evaluation
# 4. Prediction

