import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
def load_and_preprocess_data(file_path='abc_corp_financials.csv'):
    print(f"[데이터 로드] 파일 읽기: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    print(f"[데이터 로드] 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
    print(f"[데이터 로드] 날짜 범위: {df['Date'].min()} ~ {df['Date'].max()}")
    
    df.set_index('Date', inplace=True)
    
    # 결측치 확인
    missing_count = df.isnull().sum().sum()
    print(f"[전처리] 결측치 개수: {missing_count}")
    
    # 결측치 보간 (선형 보간)
    df.interpolate(method='linear', inplace=True)
    print(f"[전처리] 결측치 보간 완료 (선형 보간)")
    
    # 피처 선택 (Revenue, Cash_Inflow, Cash_Outflow -> Cash_Flow 예측)
    features = ['Revenue', 'Cash_Inflow', 'Cash_Outflow']
    target = 'Cash_Flow'
    print(f"[전처리] 입력 피처: {features}")
    print(f"[전처리] 예측 타겟: {target}")
    
    # 정규화 (MinMaxScaler)
    print(f"[전처리] 정규화 시작 (MinMaxScaler)")
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(df[features])
    scaled_target = scaler_target.fit_transform(df[[target]])
    print(f"[전처리] 정규화 완료")
    print(f"[전처리] 정규화된 피처 범위: [{scaled_features.min():.4f}, {scaled_features.max():.4f}]")
    print(f"[전처리] 정규화된 타겟 범위: [{scaled_target.min():.4f}, {scaled_target.max():.4f}]")
    
    return scaled_features, scaled_target, scaler_features, scaler_target, features, target

# 2. 시퀀스 데이터셋 생성
class TimeSeriesDataset(Dataset):
    def __init__(self, features, target, seq_length=12):
        self.features = features
        self.target = target
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.target[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# 3. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝 출력
        return out

# 4. 학습 함수
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    print(f"\n{'='*80}")
    print(f"[학습 시작] 모델 학습 준비")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[학습 설정] 디바이스: {device}")
    print(f"[학습 설정] Epoch 수: {num_epochs}")
    print(f"[학습 설정] 학습률: {lr}")
    print(f"[학습 설정] 학습 데이터 배치 수: {len(train_loader)}")
    print(f"[학습 설정] 검증 데이터 배치 수: {len(val_loader)}")
    
    model.to(device)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[학습 설정] 모델 파라미터 수: {total_params:,} (학습 가능: {trainable_params:,})")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[학습 설정] 손실 함수: MSE Loss")
    print(f"[학습 설정] 옵티마이저: Adam")
    
    train_losses, val_losses = [], []
    
    print(f"\n[학습 진행] Epoch별 손실 추이")
    print(f"{'-'*80}")
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        # 검증 모드
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 매 epoch마다 로그 출력
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}", end='')
        
        # 개선 여부 표시
        if epoch > 0:
            if val_losses[-1] < val_losses[-2]:
                print(f" | ↓ 개선 ({val_losses[-2] - val_losses[-1]:.6f})")
            elif val_losses[-1] > val_losses[-2]:
                print(f" | ↑ 악화 ({val_losses[-1] - val_losses[-2]:.6f})")
            else:
                print(f" | = 동일")
        else:
            print()
    
    print(f"{'-'*80}")
    print(f"[학습 완료] 최종 Train Loss: {train_losses[-1]:.6f}")
    print(f"[학습 완료] 최종 Val Loss: {val_losses[-1]:.6f}")
    print(f"[학습 완료] 최적 Val Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses))+1})")
    print(f"{'='*80}\n")
    
    return train_losses, val_losses

# 5. 평가 및 시각화
def evaluate_model(model, test_loader, scaler_target, device):
    print(f"\n{'='*80}")
    print(f"[평가 시작] 테스트 데이터 평가")
    print(f"{'='*80}")
    print(f"[평가 설정] 테스트 배치 수: {len(test_loader)}")
    
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            if batch_idx == 0:
                print(f"[평가 진행] 첫 번째 배치 예측 완료 (배치 크기: {len(output)})")
    
    print(f"[평가 진행] 전체 예측 완료 (총 {len(predictions)}개 샘플)")
    
    # 역정규화
    print(f"[평가 진행] 역정규화 시작")
    predictions = scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler_target.inverse_transform(np.array(actuals).reshape(-1, 1))
    print(f"[평가 진행] 역정규화 완료")
    
    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # 추가 통계
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    print(f"\n[평가 결과] 성능 지표")
    print(f"{'-'*80}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")
    print(f"R²   (R-squared):               {r2:.4f}")
    print(f"{'-'*80}")
    
    # 예측 값 범위
    print(f"\n[평가 결과] 예측 통계")
    print(f"{'-'*80}")
    print(f"실제 값 범위: [{actuals.min():.2f}, {actuals.max():.2f}]")
    print(f"예측 값 범위: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"실제 값 평균: {actuals.mean():.2f}, 표준편차: {actuals.std():.2f}")
    print(f"예측 값 평균: {predictions.mean():.2f}, 표준편차: {predictions.std():.2f}")
    print(f"{'-'*80}")
    
    # 시각화 (예측 vs 실제)
    print(f"\n[시각화] 그래프 생성 중...")
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual', linewidth=2, alpha=0.8)
    plt.plot(predictions, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
    plt.legend(fontsize=12)
    plt.title('Cash Flow Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cash Flow', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_plot.png', dpi=150)
    print(f"[시각화] 그래프 저장 완료: prediction_plot.png")
    plt.show()
    
    print(f"{'='*80}\n")

# 메인 실행
if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"LSTM 모델 학습 시작")
    print(f"{'='*80}\n")
    
    # 데이터 준비
    print(f"[단계 1/6] 데이터 로드 및 전처리")
    print(f"{'-'*80}")
    features, target, scaler_features, scaler_target, feat_names, tgt_name = load_and_preprocess_data()
    
    # 시퀀스 데이터셋 생성
    print(f"\n[단계 2/6] 시퀀스 데이터셋 생성")
    print(f"{'-'*80}")
    seq_length = 12
    print(f"[데이터셋] 시퀀스 길이: {seq_length}개월")
    dataset = TimeSeriesDataset(features, target, seq_length)
    print(f"[데이터셋] 전체 데이터셋 크기: {len(dataset)}개 샘플")
    
    # 데이터 분할 (7:2:1)
    print(f"\n[단계 3/6] 데이터 분할")
    print(f"{'-'*80}")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"[데이터 분할] 비율: Train 70% / Validation 20% / Test 10%")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    print(f"[데이터 분할] Train: {len(train_dataset)}개, Validation: {len(val_dataset)}개, Test: {len(test_dataset)}개")
    
    # DataLoader 생성
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[데이터 분할] 배치 크기: {batch_size}")
    
    # 모델 초기화
    print(f"\n[단계 4/6] 모델 초기화")
    print(f"{'-'*80}")
    model = LSTMModel(input_size=features.shape[1])
    print(f"[모델] 입력 크기: {features.shape[1]} (피처 수)")
    print(f"[모델] LSTM 레이어: 1개")
    print(f"[모델] Hidden size: 128")
    print(f"[모델] Dropout: 0.2")
    
    # 학습
    print(f"\n[단계 5/6] 모델 학습")
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # 평가
    print(f"\n[단계 6/6] 모델 평가")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_model(model, test_loader, scaler_target, device)
    
    # 모델 저장
    print(f"\n{'='*80}")
    print(f"모델 저장")
    print(f"{'='*80}")
    model_path = 'lstm_cashflow_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"[저장 완료] 모델 파일: {model_path}")
    print(f"[저장 완료] 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 추가 파일 저장
    import pickle
    scaler_path = 'scalers.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump({'features': scaler_features, 'target': scaler_target}, f)
    print(f"[저장 완료] Scaler 파일: {scaler_path}")
    
    print(f"\n{'='*80}")
    print(f"전체 프로세스 완료!")
    print(f"{'='*80}\n")