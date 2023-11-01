## 이미지 데이터에 회전, 반전, 확대/축소 등의 방법을 사용하여 데이의의 다양성을 높여줌.
import matplotlib.pyplot as plt, os, numpy as np, tqdm
from torchvision import transforms
from PIL import Image

class dataAugmentaion():
    def __init__(self, path = None):
        self.path = path
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(degrees=15),
        ])

    def augmentedImagePath(self):
        srcList = sorted(os.listdir(self.path),key=lambda x : int(x.split('_')[0]))
        
        for iName in tqdm(srcList):
            # 이미지 불러오기
            src = Image.open(self.path + iName)
            # 데이터 증강 적용
            augmented_image = self.transform(src)
            idx = int(iName.split('_')[0]) + len(srcList)//2
            newName = '_'.join([str(idx),*iName.split('_')[1:]])

            ## 이미지 저장
            augmented_image.save(f'{self.path}{newName}')
        
    def augmentationImage(self, src):
        src = Image.fromarray(src)
        return np.array(self.transform(src))
        
# if __name__ == "__main__":
#     DA = dataAugmentaion("./Train DataSet/mouse/")
#     DA.augmentedImagePath()