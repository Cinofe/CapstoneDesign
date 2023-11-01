import torch, timm, numpy as np, os, time as t, cv2, torch.nn.functional as F
from PIL import Image
from customDataset import  TRANSFORM50, TRANSFORM128, TRANSFORM244, TRANSFORM384
from Capture_Image import CaptureFrame
from threading import Thread

# 졸음 감지 클래스
class DetectSleep:
    def __init__(self, t_Type):
        s = t.time()
        # 이미지 Frame과 관심영역(ROI)를 불러올 수 있는 CaptureImage Class
        self.CF = CaptureFrame()
        print(f'Generate CF {t.time()-s:.4f}s')
        # 눈 떠있을 때 빨강, 감았을 때 초록
        self.showColor = [(0,0,255), (0,255,0)]
        # 모델 크기
        self.ModelSize = t_Type
        # 모델 타입
        self.ModelType = ['leftEye','rightEye','mouth']
        self.state = ['OPEN','CLOSE']
        # 트랜스폼 지정
        self.transforms = {'50':TRANSFORM50,'128': TRANSFORM128,'256': TRANSFORM244, '384':TRANSFORM384}
        # gpu device 사용을 위해 체크 및 로드
        self.device = self.loadGPU()
        # 비어있는 모델 구조 생성
        self.Models = self.createModel()
        # 생성된 모델 구조들에 학습된 모델 가중치 로드
        self.leftEyeModel, self.rightEyeModel, self.mouthModel = self.loadModel(self.Models)
        # 모델의 반환 결과 저장 리스트
        self.results = [None, None, None]

        self.ROI= []
        self.p1, self.p2 = [None, None, None], [None, None, None]
    
    # GPU 사용 가능 여부 확인
    def loadGPU(self):
        Useable = torch.cuda.is_available()
        print(f'Use GPU : {Useable}')
        if Useable : device = torch.device("cuda")    
        else : device = torch.device("cpu")
        return device
    
    # 모델 생성 및 초기화 모델을 GPU로 이동
    def createModel(self):
        print('Create Empty Models')
        s = t.time()
        # 비어 있는 모델 3개 생성
        models = [timm.create_model('convnext_tiny', num_classes=2, pretrained=False).to(self.device) for _ in range(3)]
        print(f'Done({t.time() - s:.4f}s)')
        return models
    
    # 저장된 모델 불러오기
    def loadModel(self, models):
        print('Loading Models')
        s = t.time()
        for i in range(3):
            models[i].load_state_dict(torch.load(f'./Models/{self.ModelType[i]}_Model_{self.ModelSize}.pth', map_location=self.device))
            # 모델 평가 모드 전환
            models[i].eval()
        print(f'Done({t.time()-s:.4f}s)')
        return models

    # 모델에 image 넣어서 결과 추출
    def getState(self):
        # 무한 반복
        while True:
            # CF에서 capture()를 통해 Frame이 정상적으로 입력되었을 때 활성화
            if not self.CF.ret:
                continue
            
            # 추출된 관심 영역이 있으면 활성화
            if not self.ROI:
                continue
            pred = [None, None, None]
            p1 = [None, None, None]
            p2 = [None, None, None]
            try:
                for i, model in enumerate([self.leftEyeModel, self.rightEyeModel, self.mouthModel]):
                    # 모델에 입력 및 결과 저장
                    with torch.no_grad():
                        src = Image.fromarray(np.uint8(self.ROI[i]))
                        inputs = self.transforms[self.ModelSize](src)
                        inputs = torch.unsqueeze(inputs, 0)
                        inputs = inputs.to(self.device)

                        outputs = model(inputs)

                        prob = F.softmax(outputs, dim=1)
                        p1[i] = round(prob.data[0, 0].item(), 5)
                        p2[i] = round(prob.data[0, 1].item(), 5)

                        _, predicted = torch.max(outputs.data, 1)
                        pred[i] = predicted.item()

            except Exception as e:
                print(e)
                continue
            self.results = pred
            self.p1 = p1
            self.p2 = p2

    # 메인 작동 함수
    def runModel(self):
        # 쓰레드를 담아 둘 리스트
        threads = []
        # 각 모델 별로 쓰레드 생성
        th = Thread(target=self.getState)
        th.start()
        threads.append(th)
        
        # CF.capture가 정상 작동할 동안 반복
        while self.CF.capture():
            s = t.time()

            self.ROI = self.CF.getROI()
            # getState의 결과 중 하나라도 누락되면 실행 안함
            if None in self.results:
                continue
            os.system('clear')
            print(f'left Eye : {self.results[0]}')
            print(f'right Eye : {self.results[1]}')
            print(f'mouth : {self.results[2]}')

            # 반복당 프레임을 측정하기 위한 FPS 계산 식
            e = t.time() - s
            print(f'fps : {e:.5f}')
    
if __name__ == "__main__":
    os.system('cls')
    DS = DetectSleep('256')
    DS.runModel()