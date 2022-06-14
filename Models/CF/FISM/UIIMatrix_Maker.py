import pandas as pd

#------User, Item 상호작용 데이터를 불러옵니다
data = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv", encoding = "utf-8") #utf-8 = 세계가 합의한 인코딩 형식
user_movie_df= data.pivot_table("time","user","item").fillna(0)

#------Timestamp 값을 1로 변경 (binary 0, 1 형식 맞추기)
def to_one(x) :
    if x != 0 :
        return 1
    else : return 0
    
print("Making User-Item Interaction Matrix...")
user_movie_df = user_movie_df.applymap(to_one) #appy, map 함수들은 열 단위로 연산이 이뤄진다. 반면 apply map은 element 단위로 연산이 된다.
print("Saving User-Item Interaction Matrix...")
user_movie_df.to_csv("user_movie_interaction.csv")
print("User-Item Interaction Matrix Saved!")