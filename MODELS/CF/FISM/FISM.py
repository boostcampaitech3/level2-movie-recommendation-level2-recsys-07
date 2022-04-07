import numpy as np
from numpy.linalg import norm 
from numpy import dot
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

#----DataFrame 불러오기
print("Loading data...")
#----만들었던 User-Interaction Matrix 불러오기
user_movie_df = pd.read_csv("/opt/ml/input/workspace/CF/Non-DL/FISM/user_movie_interaction.csv")
user_movie_matrix = user_movie_df.to_numpy() #연산 속도를 높이기 위해 사용, shape = (31360, 6807)
user_movie_matrix = user_movie_matrix[:,1:]

#----유저별 시청 영화 인덱스 2차원 배열에 저장
print("Checking watched item index by user..")
user_watched_movie = list()
for i in range(len(user_movie_matrix)) :
    user_watched_movie.append(np.nonzero(user_movie_matrix[i])) # 값이 0이 아닌 element의 인덱스 반환
print("Watched item index Complete!")

#----User Profile Vector 만들기
print("Making User profile vector...")
user_profiles = list()
cnt = 0 #유저 스스로는 빼주기 위해 counter 생성
for user in user_watched_movie :
    user_len = len(user[0])
    user_profile = np.zeros(31360) #User profile vector 초기화

    for i in range(0, len(user[0])) :
        user_profile = user_profile + user_movie_matrix[:,user[0][i]] #User profile vector에 item vector 더해주기

    user_profile = np.round(user_profile/user_len, decimals=5) #더한 item vector만큼 나눠줘서 정규화
    user_profile[cnt] = 0 # 유저 스스로에 해당하는 칸은 0으로 처리
    user_profiles.append(user_profile)
    cnt+=1
    if cnt % 1000 == 0 : # 어디까지 완료됐는지 확인
        print(f"{cnt}/31360 complete")
print("User profile vector Complete!")

#----코사인 유사도 행렬 계산
print("Calculating Cosine Similarity...")
user_profiles = np.array(user_profiles)
cosine_sim = cosine_similarity(user_profiles, np.transpose(user_movie_matrix)) #shape = (31360, 31360), (31360, 6807)
print("Cosine Simialrity Calculation Complete!")

#----이미 시청한 영화 제거
for i in range(len(user_watched_movie)) :
    watched_index = user_watched_movie[i][0]
    for index in watched_index :
        cosine_sim[i][index] = 0

#----top 10 item 결과 저장
print("Calculating Top 10 Movies to Recommend...")
result = list()
items = list(user_movie_df)[1:]
for i in range(len(cosine_sim)) :
    user_id = user_movie_df["user"][i]
    indices = np.argpartition(cosine_sim[i], -10)[-10:] # 유사도 값이 가장 큰 10개의 item의 인덱스를 추출합니다.
    for index in indices :
        result.append((user_id,int(items[index])))
print("Recommendation Calculation Complete!")

#----결과 csv 파일로 반환
print("Saving submission csv file...")
pd.DataFrame(result, columns=["user", "item"]).to_csv("submission.csv", index=False)
print("Submission file saved!")
print("All Complete!")
