
import torch.utils.data as data, os
import torchvision.transforms as transforms
from PIL import Image

class CustomDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = ['O', 'C']  # 클래스 이름을 리스트로 정의
        self.samples = []  # 데이터셋을 저장할 리스트
        
        # 클래스 이름과 해당 클래스의 이미지 파일 경로를 매핑한 딕셔너리 생성
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        # 클래스별로 이미지 파일 경로와 레이블 정보를 저장
        fnames = os.listdir(self.root)
        fnames = sorted(fnames, key= lambda x : int(x.split('_')[0]))
        for fName in fnames:
            path = self.root + fName
            item = (path, class_to_idx[fName.split('_')[-1].split('.')[0].replace('E','').replace('M','')])
            self.samples.append(item)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.samples)
    
TRANSFORM50 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop((50,50)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
TRANSFORM128 = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.CenterCrop((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
TRANSFORM244 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((244,244)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
TRANSFORM384 = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.CenterCrop((384,384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
