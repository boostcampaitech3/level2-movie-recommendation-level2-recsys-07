# level2-movie-recommendation-level2-recsys-07


## MLflow Tracking Server 정보
- MLflow는 Machine Learning 실험의 관리를 위한 정보를 기록하고 관련된 파일들을 저장, 관리하는 서버의 기능을 수행할 수 있다.
MLflow Tracking 기능을 통해 실험 관리를 위해서는 서버, 클라이언트, Artifact 버킷 세가지 환경을 다음과 같이 세팅해야한다.

![image](https://mlflow.org/docs/latest/_images/scenario_6.png)

- Mlflow Tracking 서버를 설정하는 환경 시나리오 6에서는 Remost Host(서버)를 **프록시 서버**로 활용하여 **artifact bucket**에 실험에 관련된 파일을 저장하고 **서버 내장 DB**를 활용하여 실험 로그(parameter, model, log ..)를 관리한다.

- 클라이언트에서는 Tracking Server uri를 통해 서버와 연결, 로깅 파라미터를 지정한다.

## 서버 세팅과정

1. 서버에 mlflow를 설치한다.
> **pip install mlflow==1.24** # 이전 버전에는 tracking server에 관련된 버그가 있음

2. Google Cloud Auth 패키지 설치
> pip install google-cloud-storage

3. 서버에 artifact 리소스에 접근할 수 있는 서비스 계정 키(IAM)를 등록한다.

    3.1 IAM 키 발급 
    - 참고      
        https://turtle1000.tis-tory.com/78

    3.2 키를 VM 서버로 이동 후 다음 명령어 실행
    - VM 서버에서 다음 명령어 실행(환경변수 등록)
    > export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"

4. GCP 버킷 생성  
Artifact bucket을 생성한다.  
- 참고  
    https://brunch.co.kr/@topasvga/785

5. 다음 명령어를 Virtual Machine에서 입력하여 Tracking Server 실행한다.
> mlflow server \--backend-store-uri sqlite:///mlflow.db \--artifacts-destination gs://movierec_bucket/artifacts --serve-artifacts \--host 0.0.0.0 --port 5000  

서버 접속 정보 : http://34.105.0.176:5000/

## 서버에 Tracking 하는 방법(클라이언트 코드)
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
기타 파일(모델, log 파일 등)은 `mlflow.log_artifact` 함수로 저장할 수 있다.
`with mlflow.start_run()` 블럭 내부에서 validation 이 끝난 후 모델을 저장한다
```code
    with mlflow.start_run() as run:
        #Artifact 파일 저장
        mlflow.log_artifact(<Artifact path>) #저장할 Artifact의 경로 지정
        
        #Artifact 폴더 저장
        mlflow.log_artifacts(<Artifact folder path>) #저장할 폴더를 지정하여 폴더 내 모든 파일을 저장할 수 있음
```

