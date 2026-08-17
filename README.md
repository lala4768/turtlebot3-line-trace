# Rokey 휴게소
디지털 트윈 기반 서비스 로봇 운영 시스템 구성(simulation with burger)

<br>

### Contributors
|정서윤|홍진규|나승원|이동기|
|----|----|----|----|
|**[???](https://github.com/)**|**[???](https://github.com/)**|**[lala4768](https://github.com/lala4768)**|**[???](https://github.com/)**|

<br>

## 프로젝트 개요

* **주제**
  
   휴게소 자율 주차 시스템 시뮬레이션
* **개발 배경**

  * 명절이나 연휴시 휴게소 주차에 어려움이 많음
  * 주차 공간이 협소할 경우 사고 위험이 높음
* **목표**

  * Lane Detection 및 Autorace 기술의 결합
  * 시뮬레이션 상에서 TurtleBot3의 자동주차 시스템 구현

<br>

# Flow Chart
`Default lane mode` → `traffic light`→`parking` → `Stop tunnel`

<img width="720" height="464" alt="image" src="https://github.com/user-attachments/assets/4e189172-5c6c-4343-bded-6bf15d308db7" />

<br>

## 사용 기술 및 장비

* **언어 및 환경**: Python 3.10, Ubuntu 22.04, ROS2 Humble
* **도구**: Colab, VSCode
* **로봇 하드웨어**: X
* **로봇 소프트웨어**: Gazebo

<br>

## 주요 기능

1. **차선 감지**

![lane_trace-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/317a4965-f53e-4345-9b07-f067c66c0e61)

* /camera/image_raw 토픽으로 실시간 이미지 입력
* OpenCV로 BGR 이미지를 HSV로 변환
* HSV 범위 내 픽셀만 필터링해 차선만 추출
* 추출된 픽셀을 기반으로 곡선을 추정하고 시각화


2. **장애물 회피**

* 차선 추종과 회피 모드 명령을 모두 퍼블리시
* 회피 모드가 아닐 때만, 화면 중앙과의 오차 바탕으로 선형-각속도를 계산해 퍼블리시
* 회피 모드 토픽을 받아 True면 차선 추종 멈추고 회피 모드 전환
* 회피 모드일 때만, 들어온 메시지를 즉각 퍼블리시해 장애물 회피 동작 수행


3. **신호 감지**

![traffic_light-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/79269e7d-15c2-4f49-83ff-a70b177c3a50)

* 신호 감지 노드를 실행하고, 색상 인식에 필요한 HSV 파라미터 설정
* 카메라로부터 실시간 이미지 구독하고 OpenCV로 빨간불/노란불/초록불 감지
* 신호동의 상태가 담긴 메시지를 토픽을 통해 전달
* 감지된 신호등의 상태에 따라 로봇의 주행 속도나 정지 여부 결정
* 제어 노드 실행해 실제 로봇 동작 제어


4. **표지판 감지시 주차/복귀**

![parking-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/0ec9c6ab-bcdc-4db8-98d9-c6089fbc0fa5)


* 주행 중 주차장 표지판 감지시 mux 변환
* Twist 함수 통해 주차 시작(전진 -> 좌회전 -> 전진)
* Construction sign 감지시 Twist 통해 주차 동작 실제 실행
* 주차공간 빠져나간 후 lane 복귀
* mux 재변환 후 다시 lane 따라 주행

<br>

## 프로젝트 성과

* Gazebo를 활용해 자동 주차 시스템을 시뮬레이션 환경에서 구현
* ROS2 기반 통신 구조 설계 및 노드 연동 경험

![simulation-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/8ad4640b-ad9b-459d-8325-705885b77f47)


  
