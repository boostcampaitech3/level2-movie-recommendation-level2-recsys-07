**독립적으로 실험한 모델들에 대한 앙상블을 진행하여 성능을 끌어올림**

### 1. Hard voting
- 각 모델에서 뽑은 추천 리스트에서 많이 등장한 영화를 10개 Vote

![image](https://user-images.githubusercontent.com/10546369/163725007-62f4aa3f-273a-40b4-bce3-2e90b47767a5.png)

### 2. Weighted hard voting
- 성능이 잘 나오는 모델이 추천한 영화에 가중치를 부여해서 많은 점수를 얻은 영화를 10개 Vote 

![image](https://user-images.githubusercontent.com/10546369/163725051-a38d1dae-652c-46e9-aa39-171feadd97d4.png)

### 3. Ensemble 결과

Combination| Method | Recall@10
|-----|-----|-----|
|MVAE & SASRec|Hard voting|0.1274|
|DAE & UB|Hard voting|0.1365|
|DAE & MVAE|Hard voting|0.1470|
|Top 5|Hard voting|0.1418|
|Each Model|Hard voting|0.1493|
|Top 10|Hard voting|0.1482|
|Best7|Weighted hard voting|0.1643|
|Best3|Weighted hard voting|0.1644|
|**Best4**|**Weighted hard voting**|**0.1675**|

*Best 4 : 가장 성능이 잘 나왔던 모델 4개 (Bert4Rec, Multi-VAE, DAE, UBCF)  
*Best 3 : 가장 성능이 잘 나왔던 모델 3개 (Bert4Rec, DAE, UBCF)  
*Best 7 : 가장 성능이 잘 나왔던 모델 7개 (Bert4Rec, Multi-VAE, DAE, UBCF, FFM, DeepFM,Rule Base)  
*Top 10  : 제출 성능이 가장 높았던 submission 10개  
*Top 5  : 제출 성능이 가장 높았던 submission 5개  
*Each Model : 비교적 성능이 좋았던 모델들을 겹치지 않도록 6개의 모델(Bert4Rec, SB&SASRec, DAE&MVAE, UBCF&SASRec, RecVAE, FISM&SASRec)
