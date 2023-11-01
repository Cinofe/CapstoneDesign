import torch, torch.nn as nn, torch.optim as optim, timm, torch.utils.data as data, time as t
from torch.utils.data import DataLoader
from tqdm  import tqdm
from customDataset import CustomDataset, TRANSFORM50, TRANSFORM128, TRANSFORM256

if __name__ == "__main__":
    s = t.time()

    torch.cuda.empty_cache()
    torch.no_grad()

    # 하이퍼파라미터 설정
    batch_size = 32
    num_epochs = 35
    learning_rate = 0.0001
    weight_decay = 1e-6  # L2 규제의 강도를 설정합니다.


    # 트랜스폼 지정
    transforms = {'50':TRANSFORM50,'128': TRANSFORM128,'256': TRANSFORM256}
    # 데이터셋 로드 및 전처리
    train_dataset = CustomDataset(root='./Train DataSet/MergeData/', transform=transforms['50'])
    print(f'dataSet Length : {len(train_dataset)}')

    # 모델 생성 및 초기화
    # model = timm.create_model('convnext_tiny', num_classes=6, pretrained=True)
    model = timm.create_model('mobilenetv3_rw', num_classes=6, pretrained=True)

    # 오차 함수 및 최적화 알고리즘 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # weight_decay를 추가하여 L2 규제를 적용합니다.

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 데이터셋을 GPU로 이동
    model.to(device)
    # 모델 학습 모드 전환
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    scaler = torch.cuda.amp.GradScaler()

    # 모델 학습
    for epoch in range(num_epochs):

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}")

        for i, data in enumerate(pbar):
            with torch.cuda.amp.autocast():
            # 데이터셋에서 배치 데이터와 레이블을 가져와 모델에 입력
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            pbar.set_description(desc=f"epoch {epoch+1:2}, loss: {loss:.6f}")

    # 학습된 모델 저장
    torch.save(model.state_dict(), f'Mobile_Model_50.pth')

    print(f"Finished Training\ntime : {(t.time()-s)//60:.4f}")