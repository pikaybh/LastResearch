오후 1:30 2022-08-17 | init
오후 1:31 2022-08-17 | 바탕화면의 CNN(데이터 시각화 코드) 폴더를 D:로 옮김, 시계열데이터 시각화 성공
---
오후 5:42 2022-08-18 | FPN 시각화 - blocknet? 그건 성공
오후 5:43 2022-08-18 | FPN 시각화 - p3 구현에서 non-singleton dimension 3 이게 뭔지 모르겠음
---
오후 9:30 2022-08-22 | FPN 시각화 불완전하게 구현함
---
오후 2:48 2022-08-24 | `p3에 대해 알아낸 거`: rand한 data를 넣으면 돌아가는게 lateral.size=(75, 75)이기 때문이다. 반대로 고양이 img를 넣으면 lateral.size가 (180, 180)이나, (240, 240)이 돼야 하는데 이 부분이 안 맞는다.
오후 2:53 2022-08-24 | RetinaNet()을 돌리면 생각보다 쉽게 되지 않음을 확인
---
오후 5:37 2022-08-25 | fm에 넣는 부분에서 에러 나는데 이거 왜 그런지 알아내자
---
오후 5:18 2022-09-12 | 
중간 정리 :
├─.idea
│  └─inspectionProfiles
├─data
│  ├─emojis
│  └─raw
 |─FPN
│  │  _FPN.py
│  │
│  └─graphs
├─MNIST_data
│  └─MNIST
├─Object_Localiztion_with_TensorFlow_Complete
│  │  data.bat
│  │  environment.yaml
│  │  __main__.py
│  │
│  └─__pycache__
│          __main__.cpython-39.pyc
│
├─ResNet
│  │  Blocks.py
│  │  Resnet.py
│  │  test1.py
│  │  _ResNet.py
│  │
│  ├─.idea
│  │  │  .gitignore
│  │  │  CNN.iml
│  │  │  misc.xml
│  │  │  modules.xml
│  │  │  workspace.xml
│  │  │
│  │  └─inspectionProfiles
│  │          profiles_settings.xml
│  │
│  ├─data
│  ├─graphs
│  │      epoch=1.png
│  │      epoch=1001.png
│  │      epoch=101.png
│  │      epoch=201.png
│  │      epoch=301.png
│  │      epoch=401.png
│  │      epoch=501.png
│  │      epoch=601.png
│  │      epoch=701.png
│  │      epoch=801.png
│  │      epoch=901.png
│  │
│  └─models
├─RetinaNet
│  │  model.pth
│  │  resnet50-19c8e357.pth
│  │  __main__.py
│  │
│  ├─.idea
│  │  │  .gitignore
│  │  │  misc.xml
│  │  │  modules.xml
│  │  │  RetinaNet.iml
│  │  │  workspace.xml
│  │  │
│  │  └─inspectionProfiles
│  │          profiles_settings.xml
│  │
│  ├─.pytest_cache
│  │  │  .gitignore
│  │  │  CACHEDIR.TAG
│  │  │  README.md
│  │  │
│  │  └─v
│  │      └─cache
│  │              lastfailed
│  │              nodeids
│  │              stepwise
│  │
│  └─__pycache__
│          __main__.cpython-39.pyc
│
├─tmp
│      visdom_test.py
│
└─VGG
        _VGG.py

_RetinaNet.py
Localization_RetinaNet.py

		이 중 우리의 목표는 "./Localization_RetinaNet.py"이다!

오후 5:53 2022-09-12 | 타겟을 실행시킬 기반 모듈을 TF로 할지 pyTorch로 할지 조차 모르겠다...