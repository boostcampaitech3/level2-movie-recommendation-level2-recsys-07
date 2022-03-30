
UIIMatrix_Maker.py
: 유저 아이템 상호작용 행렬을 만들어주는 행렬입니다. FISM.py를 실행하기 전에 실행해주세요!

FISM.py
: User-Item 행렬의 item vector를 사용해서 user profile vector를 만들고, 코사인 유사도를 계산해서 top 10 추천 submission.csv을 저장합니다.

submission.csv
: recall@10 값이 0.0860 나온 정답 submission 파일입니다.

user_movie_interaction.csv
: UIIMatrix_Maker.py의 결과로 나오는 행렬입니다.

user_movie_df

<img width="1099" alt="스크린샷 2022-03-30 오후 11 29 13" src="https://user-images.githubusercontent.com/96756092/160859765-a3c3714a-1eb4-4a47-af97-38e8a42fef37.png">


user_movie_matrix

<img width="248" alt="스크린샷 2022-03-30 오후 11 28 02" src="https://user-images.githubusercontent.com/96756092/160859527-16afd80e-7b26-48c5-b96c-f9de13ed3388.png">


user_profiles

<img width="498" alt="스크린샷 2022-03-30 오후 11 26 28" src="https://user-images.githubusercontent.com/96756092/160859940-efbeb264-bfb9-4fd1-b710-05ec49c2fa6c.png">


cosine_sim

<img width="558" alt="스크린샷 2022-03-30 오후 10 20 11" src="https://user-images.githubusercontent.com/96756092/160859999-b1e935b1-25f7-4c3d-8c43-29ab53c4f287.png">
