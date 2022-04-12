# EDA

### EDA 수행 결과에 대한 이미지와 간단한 설명

## 1 Total monthes
파일 이름 `EDA.ipynb`
![image](https://user-images.githubusercontent.com/41178045/159420503-0bd7fa45-16a1-4413-8ff3-68f776cd3d32.png)<br>
총 영화를 본 기간입니다<br>
사용자 별로 (마지막 영화를 본 시각)-(처음 영화를 본 시간) 의 길이를 월 단위로 나타내었습니다.<br>

## 2 인기 없는 영화
파일 이름 `[EDA]unpopular_movies.ipynb`
![b3e2ed3a-42ea-4d98-adb8-e3ade20498ff](https://user-images.githubusercontent.com/10546369/161894515-0bfb5044-6b02-412d-afdd-02479b5dc99e.png)

전체 영화의 rating 횟수에 대한 그래프

- 총 영화 : 6807개  
- 인기 없는 영화(200회 이하로 rating 된 영화) : 3423개

인기 없는 영화 를 제외하고 Inference.py 수행 시 DATA의 수는 : **101,308,131** 건 입니다.

# 이상 유저에 대한 EDA
### timestamp가 하루 이하인 user
로그 길이가 하루인 user
- 명수 : 10097
- 길이 분포<br>
![image](https://user-images.githubusercontent.com/41178045/162798935-5d14dce5-0ab5-45dc-b91c-aa9e67abb8f6.png)
### Bad users
`bad_user` = 다른 모델들로 부터 얻어진 안 좋은 결과의 user
- `bad_user`에 속하면서 로그 길이가 하루인 유저의 명수 :  4531
- `bad_user`의 log sequence 길이의 평균 : 114.88838374466506
- sequence 길이 최대 :  1571
- sequence 길이 최소 :  16
- 길이 분포<br>
![image](https://user-images.githubusercontent.com/41178045/162798966-761ecb94-d257-4769-b947-218b09ca0759.png)

### bad_user 중 이용기간이 하루 이상인 사람들
- 이용기간이 하루 이상인 bad user 중 timestamp기간이 하루보다 긴 users :  6247
- 이용기간이 하루 이상인 bad_user의 log sequence의 평균 : 134.49719865535457
- sequence 길이 최대 :  1571
- sequence 길이 최소 :  28
- 평균 기간 : 15.8개월
