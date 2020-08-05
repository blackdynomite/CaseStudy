import pandas as pd
import os
import sklearn
from sklearn import metrics
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from dtaidistance import clustering
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance


##change path appropriately##
os.chdir('C:\\Users\\Sebhat Yidelwo\\Desktop\\case4')


##data cleaning##



kale=pd.read_csv("kale_data.csv")#read dataset
list_dead=kale.index[kale['Comments'] == "Dead"].tolist()#list of row indexes with dead leaves
kale_no_dead=kale.drop(list_dead)#drop those rows



#create function which return the corresponding temperature value for each isoroom
def temperature (df):
   if df['Isoroom'] == 1:
      return 20
   if df['Isoroom'] == 2 :
      return 24
   if df['Isoroom'] == 3 :
      return 28
   if df['Isoroom'] == 4 :
      return 28
   if df['Isoroom'] == 5 :
      return 28
   if df['Isoroom'] == 6 :
      return 28
   if df['Isoroom'] == 7 :
      return 24
   if df['Isoroom'] == 8 :
      return 20

#create temperature column using temperature function
kale_no_dead['Temperature'] = kale_no_dead.apply (lambda df: temperature(df), axis=1)


#create function which return the corresponding humidity value for each isoroom
def humidity (df):
   if df['Isoroom'] == 1 :
      return 0.6
   if df['Isoroom'] == 2 :
      return 0.8
   if df['Isoroom'] == 3 :
      return 0.6
   if df['Isoroom'] == 4 :
      return 0.8
   if df['Isoroom'] == 5 :
      return 0.6
   if df['Isoroom'] == 6 :
      return 0.8
   if df['Isoroom'] == 7 :
      return 0.6
   if df['Isoroom'] == 8 :
      return 0.8

#create humidity column using temperature function
kale_no_dead['Humidity'] = kale_no_dead.apply (lambda df: humidity(df), axis=1)



#create function which return the corresponding ventilation speed value for each isoroom
def speed (df):
   if df['Isoroom'] == 1 :
      return 0.1
   if df['Isoroom'] == 2 :
      return 0.1
   if df['Isoroom'] == 3 :
      return 0.1
   if df['Isoroom'] == 4 :
      return 1
   if df['Isoroom'] == 5 :
      return 1
   if df['Isoroom'] == 6 :
      return 0.1
   if df['Isoroom'] == 7 :
      return 0.1
   if df['Isoroom'] == 8 :
      return 0.1


#create ventilation speed column using temperature function
kale_no_dead['Vent Speed'] = kale_no_dead.apply (lambda df: speed(df), axis=1)


#create column for marketable yield, waste/week and stem/week
#by dividing corresponding columns by the number of weeks until harvesting
kale_no_dead["mark yield"]=kale_no_dead['Leaves Weight']/kale_no_dead["Harvest Week"]
kale_no_dead["waste/week"]=kale_no_dead['Waste Weight']/kale_no_dead["Harvest Week"]
kale_no_dead["stem/week"]=kale_no_dead['Stem Weight']/kale_no_dead["Harvest Week"]


#drop unused columns
kale_new=kale_no_dead.drop(["Tower","PP","Date",
                                'Stem Weight','Waste Weight','Leaves Weight',"Comments"],axis=1)


#group by harvest type and conditions and find sum
#intermediate1 dataframe will be used in the end for finding the optimisation coefficients
intermediate1=kale_new.groupby(["Isoroom","Type of Harvest","Harvest Week","Temperature","Humidity","Vent Speed"]).sum()
intermediate1.to_csv(path_or_buf="intermediate1.csv")#convert to csv



#groupby again to combine the sum of partial and full harvest for each isoroom, each week
intermediate2=intermediate1.groupby(["Isoroom","Harvest Week","Temperature","Humidity","Vent Speed"]).sum()
intermediate2.to_csv(path_or_buf="final_dataset.csv")#convert to csv





kale_df=pd.read_csv("final_dataset.csv")#read final dataset



##Statistical tests for marketable yield##

#create variables to store the marketable yield for the different temp, humidity and speed conditions
sp1_yield=kale_df.loc[kale_df['Vent Speed'] == 0.1]["mark yield"]
sp2_yield=kale_df.loc[kale_df['Vent Speed'] == 1]["mark yield"]
hum80_yield=kale_df.loc[kale_df['Humidity'] == 0.8]["mark yield"]
hum60_yield=kale_df.loc[kale_df['Humidity'] == 0.6]["mark yield"]
temp20_yield=kale_df.loc[kale_df['Temperature'] == 20]["mark yield"]
temp24_yield=kale_df.loc[kale_df['Temperature'] == 24]["mark yield"]
temp28_yield=kale_df.loc[kale_df['Temperature'] == 28]["mark yield"]

#boxplot for the two speed conditions
plt.boxplot([sp1_yield,sp2_yield],labels=["sp=0.1","sp=1"])
plt.xlabel("vent speed")
plt.ylabel("marketable yield")
plt.title("marketable yield vs vent speed")
plt.show()

#boxplot for the two humidity conditions
plt.boxplot([hum80_yield,hum60_yield],labels=["hum=80","hum=60"])
plt.xlabel("humidity")
plt.ylabel("marketable yield")
plt.title("marketable yield vs humidity")
plt.show()


#boxplot for the three temperature conditions
plt.boxplot([temp20_yield,temp24_yield,temp28_yield],labels=["temp=20","temp=24","temp=28"])
plt.xlabel("temperature")
plt.ylabel("marketable yield")
plt.title("marketable yield vs temperature")
plt.show()


#mann whitney to for speed
#Ho: results come from the same population
#Ha: results come from different populations
stats.mannwhitneyu(sp1_yield,sp2_yield,alternative="two-sided")
#p-value=0.91
#cannot reject Ho


#mann whitney to for humidity
#Ho: results come from the same population
#Ha: results come from different populations
stats.mannwhitneyu(hum80_yield,hum60_yield,alternative="two-sided")
#p-value=0.66
#cannot reject Ho


#kruskal wallis test to for temperature
#Ho: results come from the same population
#Ha: at least one group comes from different population
stats.kruskal(temp20_yield,temp24_yield,temp28_yield)
#p-value=0.85
#cannot reject Ho



##Statistical tests for waste##


#create variables to store the waste for the different temp, humidity and speed conditions
sp1_waste=kale_df.loc[kale_df['Vent Speed'] == 0.1]["waste/week"]
sp2_waste=kale_df.loc[kale_df['Vent Speed'] == 1]["waste/week"]
hum80_waste=kale_df.loc[kale_df['Humidity'] == 0.8]["waste/week"]
hum60_waste=kale_df.loc[kale_df['Humidity'] == 0.6]["waste/week"]
temp20_waste=kale_df.loc[kale_df['Temperature'] == 20]["waste/week"]
temp24_waste=kale_df.loc[kale_df['Temperature'] == 24]["waste/week"]
temp28_waste=kale_df.loc[kale_df['Temperature'] == 28]["waste/week"]

plt.boxplot([sp1_waste,sp2_waste],labels=["sp=0.1","sp=1"])
plt.xlabel("vent speed")
plt.ylabel("waste/week")
plt.title("waste/week vs vent speed")
plt.show()

plt.boxplot([hum80_waste,hum60_waste],labels=["hum=80","hum=60"])
plt.xlabel("humidity")
plt.ylabel("waste/week")
plt.title("waste/week vs humidity")
plt.show()

plt.boxplot([temp20_waste,temp24_waste,temp28_waste],labels=["temp=20","temp=24","temp=28"])
plt.xlabel("temperature")
plt.ylabel("waste/week")
plt.title("waste/week vs temperature")
plt.show()


#mann whitney to for speed
#Ho: results come from the same population
#Ha: results come from different populations
stats.mannwhitneyu(sp1_waste,sp2_waste,alternative="two-sided")
#p-value=0.02
#reject Ho


#mann whitney to for humidity
#Ho: results come from the same population
#Ha: results come from different populations
stats.mannwhitneyu(hum80_waste,hum60_waste,alternative="two-sided")
#p-value=0.17
#reject Ho



#kruskal wallis test to for temperature
#Ho: results come from the same population
#Ha: at least one group comes from different population
stats.kruskal(temp20_waste,temp24_waste,temp28_waste)
#p-value=0.05
#reject Ho




#Hierarchical clustering##

#create waste/yield column by dividing waste by marketable yield
kale_df["waste/yield"]=kale_df["waste/week"]/kale_df["mark yield"]

linked = linkage(kale_df.iloc[:,2:9], 'average')#create linkage, ignoring the isoroom and harvest week columns

#labels
labels=["t20h0.6s0.1w4","t20h0.6s0.1w5","t20h0.6s0.1w6","t20h0.6s0.1w7",
        "t24h0.8s0.1w4","t24h0.8s0.1w5","t24h0.8s0.1w6","t24h0.8s0.1w7",
        "t28h0.6s0.1w4","t28h0.6s0.1w5","t28h0.6s0.1w6","t28h0.6s0.1w7",
        "t28h0.8s1w4","t28h0.8s1w5","t28h0.8s1w6","t28h0.8s1w7",
        "t28h0.6s1w4", "t28h0.6s1w5", "t28h0.6s1w6", "t28h0.6s1w7",
        "t28h0.8s0.1w4","t28h0.8s0.1w5","t28h0.8s0.1w6","t28h0.8s0.1w7",
        "t24h0.6s0.1w4","t24h0.6s0.1w5","t24h0.6s0.1w6","t24h0.6s0.1w7",
        "t20h0.8s0.1w4","t20h0.8s0.1w5","t20h0.8s0.1w6","t20h0.8s0.1w7"]

#plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,orientation='top',distance_sort='descending',
           labels=labels,
           show_leaf_counts=True,
           )
plt.xlabel("labels")
plt.ylabel("distance")
plt.title("Dendrogram")
plt.show()



#agglomerative clustering based on dendrogram clusters
clust=cluster.AgglomerativeClustering(n_clusters=3,affinity='manhattan', linkage='average')
clust.fit(kale_df.iloc[:,2:9])
L=clust.labels_
metrics.silhouette_score(kale_df.iloc[:,2:9],L)


#plot 2D graph
plt.figure(figsize=(20, 10))
for i in range(0,len(labels)):#iterate over all labels
    plt.scatter(kale_df.iloc[:,2:9]['mark yield'],kale_df.iloc[:,2:9]["waste/yield"], c=L, cmap='rainbow')#produce scatter plot with each cluster being a different colour
    plt.xlabel("mark yield")
    plt.ylabel("waste/yield")
    plt.annotate(labels[i], (kale_df.iloc[:,2:9]['mark yield'][i], kale_df.iloc[:,2:9]["waste/yield"][i]))#add the labels to the plot
    plt.axhline(y=0.1, color='red', linestyle='-',linewidth=1.5)#plot line at waste/yield=0.1
plt.title("Marketable yield vs waste/yield ratio")
plt.show()





#solve the two waste/yield and required marketable yield requirements and produce a
#plot which shows the feasibility region that satisfies both inequalties
import sympy
from sympy import *
x,y=symbols('x y')#define decision variables, x for full harvest and y for partial harvest

opt=pd.read_csv("intermediate1.csv")#read intermidiate1 csv
opt_df=opt[opt["Isoroom"]==1][["Type of Harvest","Harvest Week","mark yield","waste/week"]]

opt_df[opt_df["Harvest Week"]==5]
#optimasation coefficients
#Type of Harvest  Harvest Week  mark yield  waste/week
#  Full Harvest             5      135.06       10.12
#  Partial Harvest          5       44.14        2.44

#135.06*x+44.14*y>=210000 :   marketable yield inequality
#(10.12*x+2.44*y)/( 135.06*x+44.14*y)<=0.1 : waste inequality



full_harv_yield=135.06
partial_harv_yield=44.14
full_harv_waste=10.12
partial_harv_waste=2.44
plot_implicit(And((full_harv_yield*x+partial_harv_yield*y>=210000),
              (full_harv_waste*x+partial_harv_waste*y)/(full_harv_yield*x+partial_harv_yield*y)<=0.1),
              x_var=(x,0,5000),y_var=(y,0,5000),
              xlabel=("full harvest"),ylabel=("partial harvest"),title=("Optimisation"))

