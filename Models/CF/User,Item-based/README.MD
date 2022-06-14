# User,Item-based CF
user-item interaction matrix=
`user_item_matrix`<br>
![image](https://user-images.githubusercontent.com/41178045/161205401-fe075ca9-1fed-4de9-89e1-773373134708.png)<br>

rating이 없기 때문에 로그가 있으면 보았다는 의미로 1을 rating 대신 넣는다.<br>
user:행 item:열

## User based Collaborative Filtering
> `user_similarity` : user와 user 사이의 유사도를 계산<br>
> ![image](https://user-images.githubusercontent.com/41178045/161205481-8f92d338-9557-4227-8309-880316818db6.png)<br>
> `u_sim_top` : 각 user 당 유사도가 높은 순서로 `TOP_N`명 만큼 user_id를 가져온다.<br>
> ![image](https://user-images.githubusercontent.com/41178045/161205540-b1c5b8d7-5f27-42c8-b56f-43863c7c5e1b.png)<br>
>> TOP_N : 50으로 설정되어 있으며 hyper paramter이다.<br>
> ### 비슷한 유저들이 본 아이템 voting
> `user_item_count` : 해당 유저와 높은 유사도를 가진 다른 유저들이 본 아이템들을 세서 이 행렬에 저장한다. 기본적으로 해당 유저가 이미 본 아이템들은 추천되면 안되기 때문에 -TOP_N으로 그 값을 대체해서 아무리 더해져도 0을 넘지 않게 된다.<br>
> 결과 : Recall@10 : 0.1153 <br>

## Item based Collaborative Filter
> `item_similarity` : item과 item 사이의 유사도를 계산. 특히 여기에서는 같은 아이템 끼리의 유사도는 1이 되어 제일 먼저 추천되기 때문에 대각행렬을 0으로 만들어주어 추천에서 제외시키도록 만들었다.<br>
> ![image](https://user-images.githubusercontent.com/41178045/161205696-16779240-32cd-471b-8d53-7e2e0afd353e.png)<br>
> `per_user` : 유저가 본 item : [1,0,1,0...]<br>
> `watched_items` : `per_user`에서 1로 표시된 아이템들의 id값들만을 저장해준다. : [0,2...]<br>
> `tmp_similarity` : watched_items와 모든 items간의 유사도만들 가지고 있는 행렬이다. 이 행렬에서 가장 높은 값을 가진 item을 차례로 추천하게 된다. 결국 사용자가 이미 소비한 아이템과 모든 아이템들과의 유사도를 구해서 그 중 높은 유사도를 가진 item을 추천한다.<br>
> 결과 : Recall@10 : 0.0411<br>
