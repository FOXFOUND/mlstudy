#https://www.kaggle.com/thebrownviking20/in-depth-eda-and-clustering
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

Debug=0
DATASET_DIR='../dataset'
df=None
# pd.to_excel('output.xls')

# plt.figure(figsize=(24,12))
#
# for ar in area_list:
#     yearly_produce = []
#     for yr in year_list:
#         yearly_produce.append(df[yr][df['Area'] == ar].sum())
#     plt.plot(yearly_produce, label=ar)
# plt.xticks(np.arange(53), tuple(year_list), rotation=60)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0.)
# plt.show()
# sns.factorplot("Element", data=df, kind="count")
# sns.factorplot("Area", data=df[(df['Area'] == "India")
#                                | (df['Area'] == "China, mainland")
#                                | (df['Area'] == "United States of America")]
#                , kind="count", hue="Element", size=8, aspect=.8)
# plt.show()
new_df_dict = {}
if Debug is 1:
    df = pd.read_csv("%s/%s" % (DATASET_DIR, 'FAO.csv'), encoding='latin1')
    area_list = list(df['Area'].unique())
    year_list = list(df.iloc[:, 10:].columns)
    for ar in area_list:
        yearly_produce = []
        for yr in year_list:
            yearly_produce.append(df[yr][df['Area']==ar].sum())
        new_df_dict[ar] = yearly_produce

    np.save("new_df.npy",[new_df_dict,df])
else:
    new_df_dict = np.load("new_df.npy").tolist()[0]
    df = np.load("new_df.npy").tolist()[1]
#-----------------------------------------------------------#
#-----------------------------------------------------------#
area_list = list(df['Area'].unique())
year_list = list(df.iloc[:,10:].columns)
# print year_list
# sns.factorplot("Item", data=df, kind="count")
# plt.show()
# sys.exit()
new_df = pd.DataFrame(new_df_dict)
new_df = pd.DataFrame.transpose(new_df)
new_df.columns = year_list

mean_produce = []
for i in range(new_df_dict.__len__()):
    mean_produce.append(new_df.iloc[i].values.mean())
new_df['Mean_Produce'] = mean_produce

new_df['Rank'] = new_df['Mean_Produce'].rank(ascending=False)
new_df.head()

new_df['Mean_Produce'] = mean_produce
new_df['Rank'] = new_df['Mean_Produce'].rank(ascending=False)
# print new_df.head()
#-----------------------------------------------------------#
#-----------------------------------------------------------#

item_list = list(df['Item'].unique())

item_df = pd.DataFrame()
if Debug is 2:
    item_df['Item_Name'] = item_list
    # print item_df['Item_Name']

    for yr in year_list:
        item_produce = []
        for it in item_list:
            item_produce.append(df[yr][df['Item']==it].sum())
        item_df[yr] = item_produce

    sum_col = []
    for i in range(item_list.__len__()):
        sum_col.append(item_df.iloc[i, 2:].values.sum())
    item_df['Sum'] = sum_col
    item_df['Production_Rank'] = item_df['Sum'].rank(ascending=False)
    item_df.to_csv("item.csv")
else:
    item_df = pd.read_csv("item.csv")

# print item_df[item_df['Production_Rank'] < 11][['Item_Name','Sum']]
print item_df[['Item_Name','Sum']][item_df['Production_Rank'] < 11].sort_values(['Sum'],ascending=True)
TopTenFood = [i for i in item_df['Item_Name'][item_df['Production_Rank'] < 11]]
print TopTenFood

year_df=df.iloc[:,10:]
# year_df.corr().to_csv('corr.xls')
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(year_df.corr(),ax=ax)
#
# f, ((ax1, ax2)) = plt.subplots(1, 2, sharey='row', figsize=(10,10))
# ax1.set(xlabel='Y1968', ylabel='Y1961')
# ax2.set(xlabel='Y1968', ylabel='Y1963')
# sns.jointplot(x="Y1968", y="Y1961", data=year_df, kind="reg" , ax=ax1)
# sns.jointplot(x="Y1968", y="Y1963", data=year_df, kind="reg", ax=ax2)
# plt.close(2)
# plt.close(3)
#
# plt.show()

# new_item_df = item_df.drop(["Item_Name","Sum","Production_Rank"],axis=1)
# fig, ax = plt.subplots(figsize=(12,24))
# sns.heatmap(new_item_df,ax=ax)
# ax.set_yticklabels(item_df.Item_Name.values[::-1])
# plt.show()

X = new_df.iloc[:,:-2].values
# print new_df

X = pd.DataFrame(X)
X = X.convert_objects(convert_numeric=True)
X.columns = year_list

from sklearn.cluster import KMeans
# wcss = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
#     kmeans.fit(X)
#     print "%s %s" % (i, kmeans.inertia_)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11),wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

print X

kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)
X = X.as_matrix(columns=None)

print y_kmeans
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='Others')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='China(mainland),USA,India')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of countries by Productivity')
plt.legend()
plt.savefig('cluster.png')
plt.show()