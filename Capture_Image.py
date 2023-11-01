import cv2, mediapipe as mp, numpy as np, os, logging, itertools
from mediapipe.python.solutions.face_mesh_connections import *
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)

class CaptureFrame:
    def __init__(self, Type = None):
        if Type != None:
            self.trainPath = r""
        self.mp_face_mesh = mp.solutions.face_mesh
        
        ### 리스트 앞 부터 왼쪽 눈, 오른쪽 눈, 입 특징점 번호
        self.ROIs = [[*set(list(itertools.chain(*FACEMESH_RIGHT_EYE)))],
                     [*set(list(itertools.chain(*FACEMESH_LEFT_EYE)))],
                     [*set(list(itertools.chain(*FACEMESH_LIPS)))]]
        
        ### 비디오 영상 불러오기
        self.Cap = self.checkCap()
        ### Frame을 저장할 변수 : ndarray 타입
        self.Frame : np.ndarray
        ### Frame의 세로, 가로 크기 : int형
        self.h : int
        self.w : int

        ## 그리기용 좌표 정보
        self.Pos = [None, None, None]

        ## Frame이 있는지 체크
        self.ret = False

        self.capture()
        
    # 촬영가능한 cap을 자동으로 찾는 과정.
    def checkCap(self):
        capNum = 0
        while True:
            Cap = cv2.VideoCapture(capNum)
            if not Cap.read()[0]:
                capNum += 1
                continue
            else:
                return Cap

    ## ROI 좌표 반환
    def getPos(self):
        pos =  self.Pos.copy()
        self.Frame = cv2.cvtColor(self.Frame, cv2.COLOR_BGR2RGB)
        return pos
        
    ### Frame 추출하는 함수 : 추출 성공시 True 반환, 실패시 False 반환
    def capture(self):
        ## 비디오 영상에서 Frame 추출 : ret -> 추출 성공 :True, 실패 : False
        self.ret, self.Frame = self.Cap.read()

        ## 추출 실패시 오류 메시지 출력 후 False 반환
        if not self.ret:
            return False
        
        ## Frame 변수에 추출된 Frame의 좌우 반전 후 RGB 컬러를 BGR 컬러로 변환
        ## mediapipe가 BGR 이미지의 인식에 강함
        self.Frame = cv2.cvtColor(cv2.flip(self.Frame, 1), cv2.COLOR_RGB2BGR)

        ## Frame의 세로, 가로 값 저장
        self.h, self.w = self.Frame.shape[:2]

        ## 정상적으로 연산이 완료되면 True 반환
        return True
    
    def getROI(self):
        ## Right, Left, Mouse 특징점을 임시로 담을 리스트
        rois = []

        ## Frame의 세로, 가로 값 저장
        self.h, self.w = self.Frame.shape[:2]
        min_max = lambda x : (*x.min(axis=0), *x.max(axis=0))

        ## mediapipe Face Mesh사용
        with self.mp_face_mesh.FaceMesh(
            min_tracking_confidence = 0.9,
            refine_landmarks = True
        ) as face_mesh:
            ## Face Mesh 연산 후 landmark만 추출(여기서 평균 약 0.03초 소요)
            landMarks = face_mesh.process(self.Frame).multi_face_landmarks
            
            ## landmark가 추출되었다면 연산 진행
            ## landmark가 없다면 빈 rois 반환
            if landMarks:
                ## 추출된 landmark에서 landmark 좌표 정보 추출
                facePos = np.array([(l.x * self.w,l.y*self.h) for l in landMarks[0].landmark]).astype(int)

                ## 인식된 얼굴의 크기 구하기
                fx, fy, f_x, f_y = min_max(facePos)
                
                ## 전체 영상 크기와 인식된 얼굴 크기 비율 구하기
                ## 전체 이미지 비율과 추출된 특징 영상 크기 비율을 비교하여 실제 추출할 영역 고정 시키기
                x_ratio = int((f_x - fx)/self.w*100)
                y_ratio = int((f_y - fy)/self.h*100)

                ## 정의된 randmark 번호의 좌표 정보를 이용해
                ## 눈, 입 주변 영역 이미지 추출
                for i, roiPos in enumerate(self.ROIs):
                    # 촬영된 영상의 mediapipe 좌표에서 각 좌표상 x, y값 추출
                    roi_facePos = facePos[roiPos]
                    minX, minY, maxX, maxY = min_max(roi_facePos)
                    self.Pos[i]=((minX-10, minY-10), (maxX+10, maxY+10))
                    
                    roiFrame = self.Frame[minY-y_ratio:maxY+y_ratio, minX-x_ratio:maxX+x_ratio].copy()

                    h, w = roiFrame.shape[:2]
                    
                    if h != 50 and w != 50 and np.any(roiFrame):
                        if h < 50 or w < 50:
                            roiFrame = cv2.resize(roiFrame, dsize=(50,50),interpolation=cv2.INTER_CUBIC)
                        else : 
                            roiFrame = cv2.resize(roiFrame, dsize=(50,50),interpolation=cv2.INTER_AREA)
                    rois.append(roiFrame)
        return rois

if __name__ == "__main__":
    CF = CaptureFrame()
    import time as t
    while(CF.capture()):
        s = t.time()
        os.system('clear')
        CF.getROI()
        print(f'{t.time()-s:.5f}s')
