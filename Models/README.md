## Model 개요

```
Models
   ├── Contents-based  
   │   ├── FFM  
   │   └── DeepFM
   ├── Collaborative-filtering
   │   ├── User-based model
   │   ├── Item-based model
   │   ├── SVD model
   │   ├── Multi-VAE
   │   ├── DAE
   │   ├── RecVAE
   │   ├── BPR
   │   ├── Auto-Rec
   │   └── User Profiling
   └── Sequential
       ├── SASRec 
       ├── S3Rec
       └── Bert4Rec
```

## 모델 별 최고 성능
|Model|Augmentation/Skils|Recall10|
|-----|------------------|-------|
|DeepFM|Genre, Writer, Director Concat|0.079|
|FFM|Genre, Writer / MAE loss 적용|-|
|UBCF|Cosine similarity, voting, future item제거|0.1161|
|S3Rec|Genre|0.0892|
|Bert4Rec|Top_10_per_five inference|0.1151|
|Multi-VAE|epoch = 200, future item 제거|0.1421|
|DAE|epoch = 150, WD = 0.01|0.1420|
|Rule by Genre|Top 5 장르에 대해서  3:2:2:2:1 비율 적용|0.07|
|RecVAE|epoch = 50, gamma = 0.004|0.1243|

## 모델 선정 개요
- DeepFM : 영화에 대한 Attribute를 활용하여 사용자가 선호하는 아이템 유형을 활용해 추천하고자 선정
- FFM : DeepFM이 학습과 추론과정 소요되는 시간이 길어 가벼운 모델을 사용하고자 선정
- BERT4Rec : 유저의 시청기록을 masking 하는 clozure task가 현재 대회에서 해결하고자 하는 문제와 유사하다고 판단하여 선정하게 되었다.
- UBCF : log가 일정하지 않기 때문에 가장 단순하면서도 성능이 좋아서 baseline으로써 구현했다.
- Multi-VAE : VAE의 샘플링 기법을 활용하여 보지 않은 영화에 대해 더 정확한 추천을 하기 위해 선정하였다.
- DAE : Noise를 추가하여 학습 데이터에 과적합 되는 것을 방지하기 위해 선정하였다.
