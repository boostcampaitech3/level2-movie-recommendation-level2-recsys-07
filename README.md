# level2-movie-recommendation-level2-recsys-07


## MLflow Tracking Server 정보

다음 명령어를 Virtual Machine에서 입력하여 Tracking Server 실행
> mlflow server \--backend-store-uri sqlite:///mlflow.db \--artifacts-destination gs://movierec_bucket/artifacts --serve-artifacts \--host 0.0.0.0 --port 5000  

서버 접속 정보 : http://34.105.0.176:5000/

## 서버에 Tracking 하는 방법
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

4. Model 저장
##### TODO ####
