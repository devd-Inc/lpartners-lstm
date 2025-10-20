import pandas as pd
import numpy as np
from datetime import datetime

# 가상 데이터 생성
np.random.seed(42)  # 재현성 위해
dates = pd.date_range(start='2022-01-01', periods=36, freq='M')
revenue = np.cumsum(np.random.normal(100, 20, 36)) + 1000  # 누적 증가 트렌드
cash_in = revenue * np.random.uniform(0.8, 1.2, 36)  # 매출 기반 유입
cash_out = cash_in * np.random.uniform(0.7, 1.0, 36)  # 유출 (유입의 70-100%)
cash_flow = cash_in - cash_out  # 순 현금 흐름

# DataFrame 생성
df = pd.DataFrame({
    'Date': dates,
    'Revenue': revenue,
    'Cash_Inflow': cash_in,
    'Cash_Outflow': cash_out,
    'Cash_Flow': cash_flow
})

# 일부 결측치 추가 (테스트용)
df.loc[5:6, 'Revenue'] = np.nan
df.loc[15, 'Cash_Inflow'] = np.nan

print(df.head(10))  # 샘플 출력
df.to_csv('abc_corp_financials.csv', index=False)  # CSV 저장