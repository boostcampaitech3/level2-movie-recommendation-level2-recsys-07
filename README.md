# Movie Recommendation Competitions
![image](https://user-images.githubusercontent.com/10546369/163725289-6992b50f-6524-4cea-bdf9-66772028b577.png)


## 부스트캠프 3기 영화추천 대회 7조 유쾌한발상

<table align="center">
    <tr>
        <td align="center"><b>백승주</b></td>
        <td align="center"><b>서현덕</b></td>
        <td align="center"><b>이채원</b></td>
        <td align="center"><b>유종문</b></td>
        <td align="center"><b>김소미</b></td>
    </tr>
    <tr height="160px">
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/10546369?v=4"/>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/96756092?v=4"/>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/41178045?v=4"/>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/91870042?v=4"/>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/44887886?v=4"/>
        </td>
    </tr>
    <tr>
    </tr>
    <tr>
        <td align="center"><code>ML engineer</code></td>
        <td align="center"><code>ML Engineer</code><br><code>Data Engineer</code></td>
        <td align="center"><code>ML Engineer</code></td>
        <td align="center"><code>ML Engineer</code><br><code>Data Engineer</code></td>
        <td align="center"><code>ML Engineer</code><br><code>Data Engineer</code></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/halucinor">Github</a></td>
        <td align="center"><a href="https://github.com/davidseo98">Github</a></td>
        <td align="center"><a href="https://github.com/chae52">Github</a></td>
        <td align="center"><a href="https://github.com/killerWhale0917">Github</a></td>
        <td align="center"><a href="https://github.com/somi198">Github</a></td>
    </tr>
</table>

<br>

## 목차

- [대회 개요](#대회-개요) 
- [프로젝트 구조](#프로젝트-구조) 
- [문제 정의](#문제-정의)  
- [상세 설명](#상세-설명)    
    * [EDA](#1eda)  
    * [Data Processing](#2data-processing) 
    * [Model](#3model)   
    * [Ensemble](#4ensemble)  
- [MLflow 실험 관리](#MLflow-실험-관리)  
- [최종 순위 및 결과](#최종-순위-및-결과)

<br>


## 대회 개요

### 1. 대회 설명
전처리 된 MovieLens 데이터셋의 사용자 영화 시청 이력데이터를 바탕으로 사용자가 시청했던 영화 몇 개와 시청할 영화 몇 개를 추천한다. 단순히 sequential 예측이 아니라 중간 log가 비어있기 때문에 전반적인 유저에 대한 예측을 수행해야 한다.
추가적으로 영화에 대한 side information으로 장르, 개봉년도, 작가, 감독의 정보가 주어진다. 주어진 데이터들을 활용하여 사용자에게 10개의 영화를 추천하고 **Recall@10** 값을 평가한다.

<p align="center">
<img src='https://user-images.githubusercontent.com/10546369/163722511-cf4508bd-8a78-47d1-a49c-a4293ed60400.png' width='300'></p>

### 2. 프로젝트 기간
**대회 진행** : 2022년 3월 31일 ~ 2022년 4월 14일 19:30  
**결과 발표** : 2022년 4월 14일 19:30

<br>

## 프로젝트 구조

<details>
<summary>프로젝트구조 펼치기</summary>
<div markdown="1">

```
Project
|-- EDA
|   |-- EDA.ipynb
|   |-- README.md
|   |-- Users.ipynb
|   `-- [EDA]\ Movie\ Recommendation.ipynb
|-- Ensemble
|   |-- README.md
|   |-- hard_voting.ipynb
|   `-- soft_voting.ipynb
|-- Models
|   |-- BERT4Rec
|   |   |-- UIIM_for_MVAE.ipynb
|   |   |-- config.yaml
|   |   |-- data
|   |   |   `-- input_data.pickle
|   |   |-- dataset.py
|   |   |-- inference.ipynb
|   |   |-- inference.py
|   |   |-- loss.py
|   |   |-- model.py
|   |   |-- movierec_by_year.py
|   |   |-- movierec_neg_sample.py
|   |   |-- preprocess.py
|   |   |-- train.py
|   |   `-- utils.py
|   |-- BERT4Rec-VAE-Pytorch
|   |   |-- Data
|   |   |   `-- ml-20m
|   |   |       `-- README.txt
|   |   |-- LICENSE
|   |   |-- config.py
|   |   |-- dataloaders
|   |   |   |-- __init__.py
|   |   |   |-- ae.py
|   |   |   |-- base.py
|   |   |   |-- bert.py
|   |   |   `-- negative_samplers
|   |   |       |-- __init__.py
|   |   |       |-- base.py
|   |   |       |-- popular.py
|   |   |       `-- random.py
|   |   |-- datasets
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- ml_1m.py
|   |   |   |-- ml_20m.py
|   |   |   `-- utils.py
|   |   |-- inference.py
|   |   |-- loggers.py
|   |   |-- main.py
|   |   |-- models
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- bert.py
|   |   |   |-- bert_modules
|   |   |   |   |-- __init__.py
|   |   |   |   |-- attention
|   |   |   |   |   |-- __init__.py
|   |   |   |   |   |-- multi_head.py
|   |   |   |   |   `-- single.py
|   |   |   |   |-- bert.py
|   |   |   |   |-- embedding
|   |   |   |   |   |-- __init__.py
|   |   |   |   |   |-- bert.py
|   |   |   |   |   |-- position.py
|   |   |   |   |   |-- segment.py
|   |   |   |   |   `-- token.py
|   |   |   |   |-- transformer.py
|   |   |   |   `-- utils
|   |   |   |       |-- __init__.py
|   |   |   |       |-- feed_forward.py
|   |   |   |       |-- gelu.py
|   |   |   |       |-- layer_norm.py
|   |   |   |       `-- sublayer.py
|   |   |   |-- dae.py
|   |   |   `-- vae.py
|   |   |-- options.py
|   |   |-- requirements.txt
|   |   |-- templates.py
|   |   |-- trainers
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- bert.py
|   |   |   |-- dae.py
|   |   |   |-- utils.py
|   |   |   `-- vae.py
|   |   `-- utils.py
|   |-- BPR
|   |   |-- README.md
|   |   |-- config.py
|   |   |-- data
|   |   |   `-- user_item_reindexing.ipynb
|   |   |-- data_utils.py
|   |   |-- evaluate.py
|   |   |-- inference.ipynb
|   |   |-- main.py
|   |   |-- model.py
|   |   `-- user_item_reindexing.ipynb
|   |-- CF
|   |   |-- AutoRec
|   |   |   |-- datasets.py
|   |   |   |-- inference.py
|   |   |   |-- loss.py
|   |   |   |-- models.py
|   |   |   |-- run.ipynb
|   |   |   `-- train.py
|   |   |-- FISM
|   |   |   |-- FISM.py
|   |   |   |-- README.md
|   |   |   |-- UIIMatrix_Maker.py
|   |   |   `-- submission.csv
|   |   |-- Mult-VAE
|   |   |   |-- DAE_inference.py
|   |   |   |-- DAE_train.py
|   |   |   |-- MVAE_inference.ipynb
|   |   |   |-- README.md
|   |   |   |-- UIIM_for_MVAE.ipynb
|   |   |   |-- config.yaml
|   |   |   |-- dataloader.py
|   |   |   |-- dataset.py
|   |   |   |-- inference.py
|   |   |   |-- loss.py
|   |   |   |-- model.pt
|   |   |   |-- model.py
|   |   |   |-- train.py
|   |   |   `-- utils.py
|   |   `-- User,Item-based
|   |       |-- README.MD
|   |       |-- SVD.py
|   |       |-- UBCF_bad_users.ipynb
|   |       |-- UBCF_good_users.ipynb
|   |       |-- User,Item-based\ CF.ipynb
|   |       `-- User,Item-based_future.ipynb
|   |-- DeepFM
|   |   |-- README.md
|   |   |-- config.yml
|   |   |-- datasets.py
|   |   |-- inference.ipynb
|   |   |-- inference.py
|   |   |-- loss.py
|   |   |-- models.py
|   |   |-- preprocessing.py
|   |   |-- test.ipynb
|   |   |-- test.py
|   |   |-- train.py
|   |   `-- utils.py
|   |-- Ensemble
|   |   |-- FISM&150_submission.csv
|   |   |-- FISM&250_submission.csv
|   |   |-- FISM&RELU_submission.csv
|   |   |-- sub_concate.ipynb
|   |   `-- submission_files
|   |       |-- FISM_submission.csv
|   |       |-- RELU_submission.csv
|   |       |-- SEQ250_submission.csv
|   |       `-- SEQ_150_submission.csv
|   |-- FFM
|   |   |-- FFM.ipynb
|   |   |-- FFM_FULL.ipynb
|   |   |-- FFM_TEST.ipynb
|   |   |-- ffm_inference.py
|   |   |-- inference.ipynb
|   |   |-- model.py
|   |   |-- output
|   |   |   `-- result.md
|   |   `-- train.py
|   |-- README.md
|   |-- RuleBase
|   |   |-- bad_user_writer.ipynb
|   |   |-- clustering_genre.ipynb
|   |   |-- ensemble_good_bad.ipynb
|   |   `-- movierec_by_year.ipynb
|   `-- S3Rec
|       |-- README.md
|       |-- data
|       |-- datasets.py
|       |-- good_user_02.ipynb
|       |-- good_user_02.py
|       |-- inference.py
|       |-- models.py
|       |-- modules.py
|       |-- preprocessing.py
|       |-- run_pretrain.py
|       |-- run_train.py
|       |-- sample_submission.ipynb
|       |-- train.ipynb
|       |-- trainers.py
|       |-- trainers_euclidean_dist.py
|       `-- utils.py
|-- Preprocessing
|   |-- README.md
|   |-- bad_user.ipynb
|   |-- future.ipynb
|   |-- split_train_ratings.ipynb
|   `-- unpopular.ipynb
`-- README.md
```
</div>
</details>

<br>

## 문제 정의
### 1. 풀어야할 문제가 무엇인가?
- 유저가 볼만한 영화 추천
- 유저가 봤을 법한 영화 추천

### 2. 문제의 input, output은 무엇인가?
<p align="center">
<img src="https://user-images.githubusercontent.com/10546369/163722950-ed0a51a1-88a2-4c6e-9c03-1717a9e799fc.png" width="400"></p>

- **Input**
    - 영화 feature data - directors, writers, years, titles, genres
    - 사용하는 모델마다 다름
- **Output**
    - 각 유저마다 10개의 추천 영화

### 3. 어떻게 사용이 되는가?

- 로그가 누락된 상황에서도 사용자에게 적절한 영화가 추천되도록 한다

<br>


## 상세 설명
### 1.EDA

- [📜 EDA README](./EDA/README.md)

### 2.Data Processing

- [📜 Preprocessing README](./Preprocessing/README.md)

### 3.Model

- [📜 Model README](./Models/README.md)

### 4.Ensemble

- [📜 Ensemble README](./Ensemble/README.md)


<br>

## MLflow 실험 관리
### MLflow Tracking Server 정보

다음 명령어를 Virtual Machine에서 입력하여 Tracking Server 실행
```code
mlflow server \--backend-store-uri sqlite:///mlflow.db \--artifacts-destination gs://movierec_bucket/artifacts --serve-artifacts \--host 0.0.0.0 --port 5000  
```

서버 접속 정보 : http://34.105.0.176:5000/

### 서버에 Tracking 하는 방법
- 실험을 진행하는 클라이언트에 mlflow 설치
```code
pip install mlflow
```

1. Tracking 서버 uri 및 실험 명칭 세팅  
- 서버 정보와 실험 명칭을 세팅한다.
```code
mlflow.set_tracking_uri(<SERVER_URI>) # http://34.105.0.176:5000/
mlflow.set_experiment(<EXPRIMENT_NAME>) # 실험 이름(ex : DeepFM)
```
  
2. 기록할 Parameter 설정  
- 실험에 사용한 hyperparameter를 기록할 수 있다.  
- mlflow.log_param(string, string) 

    <details>
    <summary>참고</summary>
    <div markdown="2">

    ```code
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)
    ...
    ```
    </div>
    </details>

3. mlflow 실험 시작 및 matric 기록

- `with mlflow.start_run()` 으로 실험을 시작할 수 있다.
- with block 안에 train block과 valid block을 넣는다
- mlflow.log_metrics(dict, int) 으로 step마다 matric 기록 가능

    <details>
    <summary>참고</summary>
    <div markdown="3"> 

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
    </div>
    </details>


4. Artifacts 저장  
- 모델, log 파일, 이미지 등은 `mlflow.log_artifact` 함수로 저장할 수 있다.
- `with mlflow.start_run()` 블럭 내부에서 validation이 끝난 후 artifact를 저장한다.
    <details>
    <summary>참고</summary>
    <div markdown="3"> 

    ```code
    with mlflow.start_run() as run:
        #Artifact 파일 저장
        mlflow.log_artifact(<Artifact path>) #저장할 Artifact의 경로 지정
            
        #Artifact 폴더 저장
        mlflow.log_artifacts(<Artifact folder path>) #저장할 폴더를 지정하여 폴더 내 모든 파일을 저장할 수 있음
    ```   
    </div>
    </details>


<br>

## 최종 순위 및 결과

|리더보드|Recall@10| 순위 |
|--------|--------|-------|
|public| 0.1660 | **4등**|
|private|0.1675|**최종 3등**|

![image](https://user-images.githubusercontent.com/10546369/163723114-48d932e0-1bcb-4e0c-bc83-701b095c15e9.png)

