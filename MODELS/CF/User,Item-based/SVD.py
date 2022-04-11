import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.decomposition import TruncatedSVD
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime as dt

train_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv') 
train_df['rating']=1
user_item_matrix = train_df.pivot_table('rating', 'user', 'item').fillna(0)
user_item_matrix=user_item_matrix.astype(int)
users=user_item_matrix.index.to_list()
items=user_item_matrix.columns.to_list()

#SVD를 이용한 user latent matrix
svd=TruncatedSVD(n_components=10)
matrix=svd.fit_transform(user_item_matrix)
print("shape",matrix.shape)
corr=np.corrcoef(matrix)
print('corr.shape', corr.shape)
corr=pd.DataFrame(corr,columns=users, index=users)
user_similarity_df=corr.copy()

u_sim_top=pd.DataFrame()
TOP_N=50
for user in user_similarity_df.index:  # 3m 38s
    temp=pd.DataFrame(user_similarity_df[user].sort_values(ascending=False)[1:TOP_N+1].index,index=None).T # 0번째는 자기 자신인 1.0이라 뺌
    u_sim_top=pd.concat([u_sim_top, temp],axis=0)
u_sim_top.index=user_similarity_df.index

LEAST_VIEW=1000
view_count = user_item_matrix.sum(axis=0).to_list()
for i in range(len(items)) :
    if view_count[i] < LEAST_VIEW :
        user_item_matrix.iloc[: , i] = 3

years = pd.read_csv("/opt/ml/input/data/train/years.tsv", delimiter="\t")
id2year = dict()
for i in range(len(years["item"])) :
    id2year[years["item"][i]] = years["year"][i]

group = train_df.groupby("user")["time"].apply(max)
group = group.apply(dt.datetime.fromtimestamp)
for user in group.keys() :
    group[user] = group[user].year + 1

user_year_np = group.to_numpy()
unique_sid=user_item_matrix.columns.to_list()
uim_np = user_item_matrix.to_numpy()

id2show = dict((i, sid) for (i, sid) in enumerate(unique_sid))
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

missing_years = list()
for i in unique_sid:
    if i not in id2year.keys() :
        missing_years.append(i)
print('year 정보가 없는 item id:  ',missing_years)

# 4분 정도 소요
cnt=0
for user in range(len(uim_np)) :
    for item in range(len(uim_np[user])) : #show
        if id2show[item] in missing_years :
            cnt+=1
            continue

        item_year = id2year[id2show[item]] # show->id->year
        if item_year >= user_year_np[user] :
            uim_np[user][item] = 2
    
    if user%5000 == 0 :
        print(f"{user}/31360 users completed")

user_item_matrix=pd.DataFrame(uim_np,index=users, columns=items)

user_item_count=user_item_matrix.copy()

def to_minus(x) : # 이미 본 아이템은 추천 안 하게 하기 위해서 음수로 설정 
    if x != 0 :
        return -TOP_N
    else : return 0

user_item_count = user_item_count.applymap(to_minus) #4분 소요
user_item_count_cp=user_item_count.copy()

u_sim_top_np=u_sim_top.to_numpy()
user_item_matrix_np=user_item_matrix.to_numpy()
user_item_count_np=user_item_count_cp.to_numpy()

for i,user in enumerate(users):
    top_per_user=u_sim_top_np[i,:]
    for top in top_per_user:
        for j,item in enumerate(user_item_matrix_np[np.array(np.where(users == top))[0][0],:]) :
            if item==1:
                user_item_count_np[i,j]+=1
    if i%1000==0:
        print(i,"번째 유저까지 count 완료")

count_result=pd.DataFrame(user_item_count_np,columns=items,index=users) #numpy->dataframe->저장
pd.DataFrame(count_result).to_csv("user_svd_future_unpop.csv", index=False)

result=[]
user_item_count=pd.DataFrame(count_result)

# 3분 소요
for user in range(len(users)): # user의 id가 아닌 index로 돈다.
    for j in range(10): #top 10개 추천
        item=int(user_item_count.iloc[user,:].idxmax()) # item의 id가 들어감
        result.append([users[user],item])
        user_item_count.loc[user,str(item)]=0 # 추천했으니까 빼줌

print(result[:5])

pd.DataFrame(result, columns=["user", "item"]).to_csv("user_svd_future_unpop_submission.csv", index=False)
