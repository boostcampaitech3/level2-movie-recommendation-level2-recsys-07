# Movie Recommendation Competitions
![image](https://user-images.githubusercontent.com/10546369/163725289-6992b50f-6524-4cea-bdf9-66772028b577.png)


부스트캠프 3기 영화추천 대회 7조

[**1. 대회 개요**](#1대회-개요)  
    [1-1.대회 설명](#1-1대회-설명)  
    [1-2.프로젝트 기간](#1-2프로젝트-기간)  
    [1-3.프로젝트 구조](#1-3프로젝트-구조)  
[**2. 문제 정의**](#2문제-정의)  
    [2-1.풀어야할 문제가 무엇인가?](#2-1-풀어야할-문제가-무엇인가)  
    [2-2.문제의 input,output은 무엇인가?](#2-2-문제의-input-output은-무엇인가)  
    [2-3.어떻게 사용이 되는가?](#2-3-어떻게-사용이-되는가)  
[**3. EDA**](#3eda)     
[**4. Data Processing**](#4data-processing)    
[**5. Model**](#5model)  
    [5-1. 모델 개요](#5-1model-개요)  
    [5-2. 모델 별 최고 성능](#5-2model-별-최고-성능)  
    [5-3. 모델 선정 및 분석](#5-3모델-선정-개요)   
[**6. Ensemble**](#6ensemble앙상블)  
    [6-1 Hard voting](#6-1-hard-voting)  
    [6-2 Weighted hard voting](#6-2-weighted-hard-voting)  
    [6-3. Ensemble 결과](#6-3-ensemble-결과)  
[**7. mlflow 실험 관리**](#7-mlflow-실험-관리)  
[**8. 최종 순위 및 결과**](#8-최종-순위-및-결과)


# 1.대회 개요

## 1-1.대회 설명
전처리 된 MovieLens 데이터셋의 사용자 영화 시청 이력데이터를 바탕으로 사용자가 시청했던 영화 몇 개와 시청할 영화 몇 개를 추천한다. 단순히 sequential 예측이 아니라 중간 log가 비어있기 때문에 전반적인 유저에 대한 예측을 수행해야 한다.
추가적으로 영화에 대한 side information으로 장르, 개봉년도, 작가, 감독의 정보가 주어진다. 주어진 데이터들을 활용하여 사용자에게 10개의 영화를 추천하고 **Recall@10** 값을 평가한다.

<p align="center">
<img src='https://user-images.githubusercontent.com/10546369/163722511-cf4508bd-8a78-47d1-a49c-a4293ed60400.png' width='500'></p>

## 1-2.프로젝트 기간
**대회 진행** : 2022년 3월 31일 ~ 2022년 4월 14일 19:30  
**결과 발표** : 2022년 4월 14일 19:30

## 1-3.프로젝트 구조

<details>
<summary>프로젝트구조 펼치기</summary>
<div markdown="1">

```
Project
├── EDA
│   ├── EDA.ipynb
│   ├── README.md
│   └── [EDA] Movie Recommendation.ipynb
├── MODELS
│   ├── BERT4Rec
│   │   ├── UIIM_for_MVAE.ipynb
│   │   ├── config.yaml
│   │   ├── dataset.py
│   │   ├── inference.ipynb
│   │   ├── inference.py
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── BERT4Rec-VAE-Pytorch
│   │   ├── Data
│   │   │   └── ml-20m
│   │   │       └── README.txt
│   │   ├── Images
│   │   │   ├── ML1m-results.png
│   │   │   ├── ML20m-results.png
│   │   │   └── vae_tensorboard.png
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── config.py
│   │   ├── dataloaders
│   │   │   ├── __init__.py
│   │   │   ├── ae.py
│   │   │   ├── base.py
│   │   │   ├── bert.py
│   │   │   └── negative_samplers
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       ├── popular.py
│   │   │       └── random.py
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── ml_1m.py
│   │   │   ├── ml_20m.py
│   │   │   └── utils.py
│   │   ├── experiments
│   │   │   └── test_2022-04-08_0
│   │   │       └── config.json
│   │   ├── loggers.py
│   │   ├── main.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── bert.py
│   │   │   ├── bert_modules
│   │   │   │   ├── __init__.py
│   │   │   │   ├── attention
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── multi_head.py
│   │   │   │   │   └── single.py
│   │   │   │   ├── bert.py
│   │   │   │   ├── embedding
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── bert.py
│   │   │   │   │   ├── position.py
│   │   │   │   │   ├── segment.py
│   │   │   │   │   └── token.py
│   │   │   │   ├── transformer.py
│   │   │   │   └── utils
│   │   │   │       ├── __init__.py
│   │   │   │       ├── feed_forward.py
│   │   │   │       ├── gelu.py
│   │   │   │       ├── layer_norm.py
│   │   │   │       └── sublayer.py
│   │   │   ├── dae.py
│   │   │   └── vae.py
│   │   ├── options.py
│   │   ├── requirements.txt
│   │   ├── run.ipynb
│   │   ├── templates.py
│   │   ├── trainers
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── bert.py
│   │   │   ├── dae.py
│   │   │   ├── utils.py
│   │   │   └── vae.py
│   │   └── utils.py
│   ├── CF
│   │   ├── AutoRec
│   │   │   ├── datasets.py
│   │   │   ├── inference.py
│   │   │   ├── loss.py
│   │   │   ├── models.py
│   │   │   ├── run.ipynb
│   │   │   └── train.py
│   │   ├── FISM
│   │   │   ├── FISM.py
│   │   │   ├── README.md
│   │   │   ├── UIIMatrix_Maker.py
│   │   │   └── submission.csv
│   │   ├── Mult-VAE
│   │   │   ├── DAE_inference.py
│   │   │   ├── DAE_train.py
│   │   │   ├── MVAE_inference.ipynb
│   │   │   ├── README.md
│   │   │   ├── UIIM_for_MVAE.ipynb
│   │   │   ├── dataloader.py
│   │   │   ├── dataset.py
│   │   │   ├── loss.py
│   │   │   ├── model.pt
│   │   │   ├── model.py
│   │   │   └── train.py
│   │   └── User,Item-based
│   │       ├── README.MD
│   │       └── User,Item-based CF.ipynb
│   ├── DeepFM
│   │   ├── README.md
│   │   ├── config.yml
│   │   ├── datasets.py
│   │   ├── inference.ipynb
│   │   ├── inference.py
│   │   ├── loss.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   ├── test.ipynb
│   │   ├── test.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── Ensemble
│   │   ├── FISM&150_submission.csv
│   │   ├── FISM&250_submission.csv
│   │   ├── FISM&RELU_submission.csv
│   │   ├── sub_concate.ipynb
│   │   └── submission_files
│   │       ├── FISM_submission.csv
│   │       ├── RELU_submission.csv
│   │       ├── SEQ250_submission.csv
│   │       └── SEQ_150_submission.csv
│   └── RuleBase
│       └── movierec_by_year.ipynb
├── README.md
└── feature
    ├── split_train_ratings.ipynb
    └── trainers.py
```
</div>
</details>

## 2.문제 정의
### 2-1. 풀어야할 문제가 무엇인가?
- 유저가 볼만한 영화 추천
- 유저가 봤을 법한 영화 추천
- 어느 위치에 있는 영화를 예측해야 하는지 알 수 없다.
- 과거에 봤던 것일지도 모르고 미래에 볼 것인지도 모르고
- 성향이 시간에 따라 변하는 사람
    - 한 달 이내 - 그 이상 그룹으로 나눠서 다른 모델 적용
    그 사람이 들어온 해에서 유명했던 아이템 ( 그 전 1년동안)
    - 영화 200개 이내로 본 그룹 & 200이상 본 그룹 다른 모델
- 2010년에 이미 사용을 끝낸 유저에게는 2014년 영화 배제해야함
- Temporal Split적용

### 2-2. 문제의 input, output은 무엇인가?

![image](https://user-images.githubusercontent.com/10546369/163722950-ed0a51a1-88a2-4c6e-9c03-1717a9e799fc.png)

- **Input**
    - train_ratings.csv
    - 영화 feature data - directors, writers, years, titles, genres
        - titles 을 어떻게 사용할지 정해야함
- **Output**
    - 각 유저마다 10개의 추천 영화
### 2-3 어떻게 사용이 되는가?
- **로그가 누락된 상황에서도 사용자에게 적절한 영화가 추천되도록 만들어야한다**


## 3.EDA

- [📜 EDA README](./EDA/README.md)


## 4.Data Processing

**1. Future item 제거**

![image](https://user-images.githubusercontent.com/10546369/163724451-721d2554-b31b-4cb2-add0-675c7f1f376e.png)

사용자의 마지막 timestemp 이후에 개봉된 영화는 못 볼 것이기 때문에, 마지막 이용년도 +2 이후 영화는 추천대상에서 제외 함  

**2. Unpopular item 제거**

![image](https://user-images.githubusercontent.com/10546369/161894515-0bfb5044-6b02-412d-afdd-02479b5dc99e.png)

전체 영화의 rating 정보를 이용해 인기 없는 영화(시청횟수 기준)를 제거하여 사용자에게 추천하지 않도록 전처리 함


|제거 기준|200번 이하|500번 이하|1000번 이하|1200번 이하|1500번 이하|1750번 이하|2000번 이하|3000번 이하|4000번 이하|5000번 이하|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|제거 비율|50.3%| 71.2%| 84.3%| 85.2%| 87.8%| 89.4%| 91%| 94.1%| 96.1%| 97%|

## 5.Model
### 5-1.Model 개요

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

### 5-2.모델 별 최고 성능
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

### 5-3.모델 선정 개요
- DeepFM : 영화에 대한 Attribute를 활용하여 사용자가 선호하는 아이템 유형을 활용해 추천하고자 선정
- FFM : DeepFM이 학습과 추론과정 소요되는 시간이 길어 가벼운 모델을 사용하고자 선정
- BERT4Rec : 유저의 시청기록을 masking 하는 clozure task가 현재 대회에서 해결하고자 하는 문제와 유사하다고 판단하여 선정하게 되었다.
- UBCF : log가 일정하지 않기 때문에 가장 단순하면서도 성능이 좋아서 baseline으로써 구현했다.
- Multi-VAE : VAE의 샘플링 기법을 활용하여 보지 않은 영화에 대해 더 정확한 추천을 하기 위해 선정하였다.
- DAE : Noise를 추가하여 학습 데이터에 과적합 되는 것을 방지하기 위해 선정하였다.

## 6.Ensemble(앙상블)
**독립적으로 실험한 모델들에 대한 앙상블을 진행하여 성능을 끌어올림**

### 6-1. Hard voting
- 각 모델에서 뽑은 추천 리스트에서 많이 등장한 영화를 10개 Vote

![image](https://user-images.githubusercontent.com/10546369/163725007-62f4aa3f-273a-40b4-bce3-2e90b47767a5.png)

### 6-2. Weighted hard voting
- 성능이 잘 나오는 모델이 추천한 영화에 가중치를 부여해서 많은 점수를 얻은 영화를 10개 Vote 

![image](https://user-images.githubusercontent.com/10546369/163725051-a38d1dae-652c-46e9-aa39-171feadd97d4.png)

### 6-3. Ensemble 결과

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

## 7. MLflow 실험 관리
### MLflow Tracking Server 정보

다음 명령어를 Virtual Machine에서 입력하여 Tracking Server 실행
> mlflow server \--backend-store-uri sqlite:///mlflow.db \--artifacts-destination gs://movierec_bucket/artifacts --serve-artifacts \--host 0.0.0.0 --port 5000  

서버 접속 정보 : http://34.105.0.176:5000/

### 서버에 Tracking 하는 방법
- 실험을 진행하는 클라이언트에 mlflow 설치
> pip install mlflow

1. Tracking 서버 uri 및 실험 명칭 세팅  
서버 정보와 실험 명칭을 세팅한다.
```code
mlflow.set_tracking_uri(<SERVER_URI>) # http://34.105.0.176:5000/
mlflow.set_experiment(<EXPRIMENT_NAME>) # 실험 이름(ex : DeepFM)
```
  
2. 기록할 Parameter 설정  
실험에 사용한 hyperparameter를 기록할 수 있다.  
- mlflow.log_param(string, string) 

- 참고
```code
mlflow.log_param("seed", args.seed)
mlflow.log_param("epochs", args.epochs)
mlflow.log_param("batch_size", args.batch_size)
...
```

3. mlflow 실험 시작 및 matric 기록

`with mlflow.start_run()` 으로 실험을 시작할 수 있다.
- with block 안에 train block과 valid block을 넣는다
- mlflow.log_metrics(dict, int) 으로 step마다 matric 기록 가능

- 참고
```code
with mlflow.start_run() as run:
    #train block
    #train 관련 코드 입력
    mlflow.log_metrics(
        <dict>, # 기록하고 싶은 matric((ex : loss, accuracy))
        step    # 현재 step(epoch or train step)
    )
    #valid block
    #validation 관련 코드
    mlflow.log_metrics(
        <dict>, # 기록하고 싶은 matric((ex : loss, accuracy))
        step    # 현재 step(epoch or train step)
    )
```

4. Artifacts 저장  
모델, log 파일, 이미지 등은 `mlflow.log_artifact` 함수로 저장할 수 있다.
- `with mlflow.start_run()` 블럭 내부에서 validation이 끝난 후 artifact를 저장한다.
```code
with mlflow.start_run() as run:
     #Artifact 파일 저장
     mlflow.log_artifact(<Artifact path>) #저장할 Artifact의 경로 지정
        
     #Artifact 폴더 저장
     mlflow.log_artifacts(<Artifact folder path>) #저장할 폴더를 지정하여 폴더 내 모든 파일을 저장할 수 있음
```   


## 8. 최종 순위 및 결과

|리더보드|Recall@10| 순위 |
|--------|--------|-------|
|public| 0.1660 | **4등**|
|private|0.1675|**최종 3등**|

![image](https://user-images.githubusercontent.com/10546369/163723114-48d932e0-1bcb-4e0c-bc83-701b095c15e9.png)

