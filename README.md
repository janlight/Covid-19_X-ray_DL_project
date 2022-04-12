# Covid-19_X-ray_DL_project

# <<u>Covid-19 X-ray Model</u>>

# [<u>Introduction</u>]

- AI 기술은 현재 많은 곳에 접목되어 그 우수성을 증명하고 있다. 그 중  의료분야에서도 AI 기술은 많은 이슈와 발전가능성을 보이고 있다. 

- 현재 전세계가 코로나 바이러스로 인하여 많은 패닉에 빠져있다. 급속한 확산으로 인하여 세계보건기구(WHO)는 펜데믹을 선언했다. 계속되는 코로나 감염자들로 인해 의료현장에서는 많은 인력과 병상 부족을 겪고있다. 

- 현재에도 코로나 pcr 검사라는 뛰어는 기술이 있지만 무증상자 혹은 다른 질병으로 병원을 방문 할 시 x-ray 검사를 활용하여 자동적으로 코로나 검사를 실시하여 2차적인 감염의 피해를 막을 수 있지 않을까 하는 생각에서 이 프로젝트를 시작하였다.

- 또한 코로나는 호흡기 계통 통해 감염되는데 그 증상 중 하나인 호흡기 질병 중에 폐렴으로 악화되는 경우가 많고 이는 사람의 생명에 치명적으로 악영향을 끼칠 수 있다. 

- 의료 분야에서 x-ray를 직접 보고 코로나 질환의 유무를 판단하는 절차와 정확도를 간소화하고 높이기 위해 의료데이터를 활용하여 추가적인 감염과 의료 현장에 도움이 될 수 있는 AI 모델을 구축하기로 결정했다. 


# **[Data]**

![image](https://user-images.githubusercontent.com/89772868/162955987-512b0a09-282d-4246-84bd-16b2bce7987c.png)

- 사용 데이터로는 Kaggle의 dataset이용 ([https://www.kaggle.com/ahemateja19bec1025/covid-xray-dataset](https://www.kaggle.com/ahemateja19bec1025/covid-xray-dataset))
- 데이터는 약 3000개의 이미지로 구성되어 있다.
- Without Covid(0), With Covid(1)로 라벨링되어있다.

# [<u>hypothesis</u>]
![image](https://user-images.githubusercontent.com/89772868/162235156-dcd36d37-d588-43b0-beb5-935c34fe4675.png)

- X-ray 사진의 음영을 비교하여 코로나 바이러스로 발생한 폐렴의 증상을 구분할 수 있을 것이다.  

- 딥러닝 모델을 활용하여 사진의 특성을 추출해 정상 흉부 사진과 코로나로 인해 감염된 흉부 사진을 분류할 수 있을 것이다. 

- 사전 학습 모델(Pre-trained Model)을 활용하여 높은 성능의 딥러닝 모델을 구축할 수 있을 것이다.


# **[**Project process**]**

- 딥러닝 모델에 사용하기 위한 데이터 전처리를 수행한다.

- 하이퍼파라미터튜닝을 수행하여 사전 학습 모델인 VGG16, ResNet50, NASNetLarge와 완전 연결층(fully connected layer)을 구성하고 딥러닝 모델을 구축한다.

- VGG16, ResNet50, NASNetLarge을 사용한 결과를 비교하여 최종 모델을 선정한다.

- 최종 선정된 Covid-19 분류 모델의 결과를 분석한다.
  
  
# [<u>preprocessing</u>]

  ![image](https://user-images.githubusercontent.com/89772868/162236413-e6e4e7a5-c4b6-4570-90fd-874fb081c9da.png)

 
- 이미지 크기: 224x224로 맞춘다.

- 각각의 픽셀 값은 0~1 사이 값으로 정규화 해준다.

- 3091개의 데이터셋을 무작위로 섞는다.(random shuffle)

- 데이터셋을 64:16:20의 train, val, test로 나눈다.

- 0(정상 흉부 사진)과 1(코로나 감염자 흉부 사진)로 나누어진 폴더 사진들을 non-COVID와 COVID-19로 라벨링 한다.  
  
  <br></br>
  ![image](https://user-images.githubusercontent.com/89772868/162236476-011ee92a-7094-4547-9911-2e7fa9277c55.png)

- 무작위로 출력하여 전처리 된 이미지를 확인할 수 있다.

  <br></br>
## **[사전 학습 모델(Pre-trained Model)]**

![image](https://user-images.githubusercontent.com/89772868/162957181-5e5bcb96-1eca-4d3a-a98a-e63b7b773b41.png)

- ImageNet 이미지 분류에서 NASNet은 검증 세트에서 82.7%의 예측 정확도 보여준다.

- NASNet은 매우 낮은 계산 비용으로 우수한 정확도를 달성하는 일련의 모델을 생성하도록 크기를 조정할 수 있다.

- 이미지 분류에 많이 사용 되는 VGG16, ResNet50 모델을 사용하여 NASNet과 성능을 비교한다.


## [<u>완전 연결 층(fully connected layer)</u>]

![image](https://user-images.githubusercontent.com/89772868/162237223-0907a043-049d-4225-9c25-7d287da0267e.png)

- 완전 연결 층의 Dense층은 각각 512, 256개의 노드로 이루어져 있다.

- 또한 Dense층 사이에 배치 정규화(BatchNormalization)와 Dropout 층으로 구성하여 모델이 일반화를 이루게 한다.

- 각각의 Dense의 활성화 함수는 relu 함수로 구성되어있고 출력층의 활성화 함수는 softmax 함수 이다.


# [<u>hypothesis test</u>]
  
- 사전 학습 모델인 VGG16, ResNet50, NASNetLarge과 완전 연결 층(fully connected layer)으로 모델을 구축한다.

- 각각의 사전 학습 모델과 완전 연결 층을 비교 하기 위해 EarlyStopping callbacks 함수, Epochs, batch size, optimizer는 동일하게 적용한다. 

- ModelCheckpoint callbacks  함수를 사용해 모델의 개선 여부를 판단한다.

- 학습률 스케쥴러로서, 학습률의 최대값과 최소값을 정해서 그 범위의 학습률을 코사인 함수를 이용하여 스케쥴링하는 방법인 코사인 어닐링(Cosine annealing)을 사용한다.

- 성능 평가 지표는 accuracy와 f1 score를 확인한다. (Chance Level은 0.5 이다.)



## [<u>VGG16</u>]

- VGG16 모델의 train과 validation의 loss와 accuracy는 아래와 같이 나타난다.

![image](https://user-images.githubusercontent.com/89772868/162976198-837021dd-6bf3-4a5f-8628-2c0525e46486.png)

<br></br>
- VGG16모델의 test accuracy는 약 0.5703으로 나타난다.
- Non-COVID-19를 감지하기 위한 f1 score(0.73)으로 나타난다.
- VGG16 의 모델의 결과 COVID-19 X-ray 이미지를 감지하는데 낮은 성능을 나타낸다.

![image](https://user-images.githubusercontent.com/89772868/162976499-0712ab51-02d7-445f-b737-a70b7e0b4184.png)

<br></br>
## [<u>ResNet50</u>]
- ResNet50 모델의 train과 validation의 loss와 accuracy는 아래와 같이 나타난다.

![image](https://user-images.githubusercontent.com/89772868/162976703-898124f1-c50c-4e10-8bdc-dd11786e9994.png)

<br></br>
- ResNet50모델의 test accuracy는 약 0.4297으로 나타난다.
- COVID-19를 감지하기 위한 f1 score(0.60)으로 나타난다.
- ResNet50의 모델의 결과 COVID-19 X-ray 이미지를 감지하는데 낮은 성능을 나타낸다.

![image](https://user-images.githubusercontent.com/89772868/162976955-453b6181-ba9d-406e-8996-5be3e0531495.png)

<br></br>
## [<u>NASNetLarge</u>]

- NASNetLarge 모델의 train과 validation의 loss와 accuracy는 아래와 같이 나타난다.

![image](https://user-images.githubusercontent.com/89772868/162977061-4434a91d-508f-4c93-846a-bbc8c21037ad.png)

<br></br>
- NASNetLarge모델의 test accuracy는 약 0.9289로 나타난다.
- COVID-19를 감지하기 위한 f1 score(0.91)로 다음과 같이 나타난다.
- NASNetLarge모델의 결과COVID-19 X-ray 이미지를 감지하는데 우수한 성능을 나타나는 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/89772868/162977259-398d95d9-056a-4cdc-a4c8-e70b4e103ddf.png)

<br></br>
# [<u>Model test</u>]

- 모델 훈련 후 test set의 이미지를 무작위로 모델에 적용하여 예측한 시각화 자료이다.

- 작은 이미지 하단에 표시된 "I"는 이미지 인덱스, "P"는 예측값, "L"은 실제 레이블 값으로 NASNetLarge가 VGG16과 ResNet50 보다 높은 정확성을 보이는 것을 확인 할 수 있다.(글자의 초록색은 예측값과실제 레이블이 동일할 때 표현된다. 다르게 되면 빨간색으로 표현된다.)

![image](https://user-images.githubusercontent.com/89772868/162977648-56ce1c12-a02e-4a07-a76c-3a567f79de94.png)

# [<u>최종모델 선정</u>]
-- NASNetLarge 구조 --
- 최종적으로 사전 훈련 모델을 비교한 결과 NASNetLarge가 가장 성능이 우수한 모델로 나타났다.
- 전체적인 모델의 구조는 다음과 같다.

![image](https://user-images.githubusercontent.com/89772868/162239634-19cb204a-bdac-487c-9b72-8dec091b2d58.png)



# [<u>References</u>]
- AutoML for large scale image classification and object detectionhttps://ai.googleblog.com/2017/11/automl-for-large-scale-image.html

- Accurate Detection of COVID-19 Using K-EfficientNet Deep Learning Image Classifier and K-COVID Chest X-Ray Images Dataset https://ieeexplore.ieee.org/document/9344949

- Deep Convolutional Neural Network–Based Computer-Aided Detection System for COVID-19 Using Multiple Lung Scans: Design and Implementation Study https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0242759

- Deep Learning COVID-19 Features on CXR using Limited Training Data Sets Yujin Oh1 , Sangjoon Park1 , and Jong Chul Ye, Fellow, IEEE https://arxiv.org/pdf/2004.05758.pdf

- https://ai4nlp.tistory.com/16

- https://sh-tsang.medium.com/review-nasnet-neural-architecture-search-network-image-classification-23139ea0425d

- https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html



