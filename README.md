# 데이터 전처리 가이드

다양한 형식의 재무 데이터를 `abc_corp_financials` 형식으로 변환하는 전처리 도구입니다.

## 📁 파일 구조

```
lpartners-lstm/
├── create_dataset.py              # 원본 데이터 생성 스크립트
├── preprocess_data.py             # 전처리 함수 및 예시 데이터 생성
├── example_usage.py               # 사용 예시 코드
├── abc_corp_financials.csv        # 표준 형식 데이터
├── example_korean_columns.csv     # 예시: 한국어 컬럼명
├── example_different_date_format.csv  # 예시: 다른 날짜 형식
├── example_monthly_values.csv     # 예시: 월별 순증가값
├── example_string_numbers.csv     # 예시: 문자열 숫자
└── example_extra_columns.csv      # 예시: 추가 컬럼
```

## 🎯 표준 형식 (abc_corp_financials)

```csv
Date,Revenue,Cash_Inflow,Cash_Outflow,Cash_Flow
2022-01-31,1109.93,1026.34,820.32,206.02
2022-02-28,1207.17,1216.86,875.00,341.86
...
```

### 컬럼 설명

- **Date**: 날짜 (YYYY-MM-DD 형식)
- **Revenue**: 매출 (누적값 또는 월별값)
- **Cash_Inflow**: 현금 유입
- **Cash_Outflow**: 현금 유출
- **Cash_Flow**: 순 현금 흐름 (자동 계산)

## 🚀 빠른 시작

### 1. 라이브러리 설치

```bash
# 가상환경 활성화
source env/bin/activate

# 필요한 라이브러리 설치 (이미 설치됨)
pip install pandas numpy
```

### 2. 예시 데이터 생성 및 확인

```bash
# 예시 데이터 생성
python preprocess_data.py

# 사용 예시 실행
python example_usage.py
```

## 📖 전처리 함수 사용법

### 기본 사용법

```python
import pandas as pd
from preprocess_data import preprocess_to_abc_format

# 데이터 읽기
df = pd.read_csv('your_data.csv')

# 전처리 실행
result = preprocess_to_abc_format(df)

# 결과 저장
result.to_csv('converted_data.csv', index=False)
```

### 커스텀 컬럼 매핑

```python
# 한국어 컬럼명이 있는 경우
config = {
    'date_col': '날짜',
    'revenue_col': '매출',
    'cash_in_col': '현금유입',
    'cash_out_col': '현금유출'
}

result = preprocess_to_abc_format(df, config)
```

### 다른 날짜 형식 처리

```python
# DD/MM/YYYY 형식
config = {
    'date_col': 'transaction_date',
    'revenue_col': 'sales',
    'cash_in_col': 'inflow',
    'cash_out_col': 'outflow',
    'date_format': '%d/%m/%Y'  # 날짜 형식 지정
}

result = preprocess_to_abc_format(df, config)
```

## 📋 지원하는 입력 형식

### 1. 한국어 컬럼명

```csv
날짜,매출,현금유입,현금유출
2023-01-31,1102.07,857.30,703.48
```

### 2. 다른 날짜 형식

```csv
transaction_date,sales,inflow,outflow
31/01/2023,1103.04,819.98,958.72
```

### 3. 월별 순증가값

```csv
period,monthly_revenue,monthly_inflow,monthly_outflow
2023-01-31,95.10,907.23,998.84
```

### 4. 문자열 숫자

```csv
date,revenue,inflow,outflow
2023-01-31,1085.71,1052.87,795.93
```

### 5. 추가 컬럼이 있는 데이터

```csv
Date,Revenue,Cash_Inflow,Cash_Outflow,Company_ID,Region
2023-01-31,1102.39,1082.67,999.54,ABC_CORP,Seoul
```

## 🔧 전처리 함수 파라미터

### `preprocess_to_abc_format(df, config=None)`

#### Parameters

- **df**: pandas.DataFrame - 입력 데이터프레임
- **config**: dict (선택) - 변환 설정
  - `date_col`: 날짜 컬럼명 (기본값: 'Date')
  - `revenue_col`: 매출 컬럼명 (기본값: 'Revenue')
  - `cash_in_col`: 현금유입 컬럼명 (기본값: 'Cash_Inflow')
  - `cash_out_col`: 현금유출 컬럼명 (기본값: 'Cash_Outflow')
  - `date_format`: 날짜 형식 (기본값: '%Y-%m-%d')

#### Returns

- **pandas.DataFrame** - 표준 형식으로 변환된 데이터프레임

## 💡 주요 기능

### 1. 자동 컬럼 매핑

다양한 컬럼명을 표준 형식으로 자동 변환

### 2. 날짜 형식 변환

- YYYY-MM-DD
- DD/MM/YYYY
- YYYY-MM (년-월)
- 기타 strftime 형식

### 3. 데이터 타입 자동 변환

- 문자열 숫자 → float64
- 날짜 문자열 → datetime64

### 4. 결측치 처리

- 날짜가 없는 행 자동 제거
- 숫자 결측치는 유지 (NaN)

### 5. 순 현금 흐름 자동 계산

```python
Cash_Flow = Cash_Inflow - Cash_Outflow
```

## 📝 사용 예시

### 예시 1: 한국어 데이터 변환

```python
import pandas as pd
from preprocess_data import preprocess_to_abc_format

# 한국어 컬럼명 데이터 읽기
df = pd.read_csv('example_korean_columns.csv')

# 컬럼 매핑 설정
config = {
    'date_col': '날짜',
    'revenue_col': '매출',
    'cash_in_col': '현금유입',
    'cash_out_col': '현금유출'
}

# 변환 실행
result = preprocess_to_abc_format(df, config)

# 결과 확인
print(result.head())
print(result.dtypes)

# 저장
result.to_csv('converted_data.csv', index=False)
```

### 예시 2: 다른 날짜 형식 변환

```python
# DD/MM/YYYY 형식 데이터
df = pd.read_csv('example_different_date_format.csv')

config = {
    'date_col': 'transaction_date',
    'revenue_col': 'sales',
    'cash_in_col': 'inflow',
    'cash_out_col': 'outflow',
    'date_format': '%d/%m/%Y'
}

result = preprocess_to_abc_format(df, config)
```

### 예시 3: 월별 데이터를 누적값으로 변환

```python
# 월별 순증가값 데이터
df = pd.read_csv('example_monthly_values.csv')

# 누적값으로 변환
df['monthly_revenue'] = df['monthly_revenue'].cumsum()

config = {
    'date_col': 'period',
    'revenue_col': 'monthly_revenue',
    'cash_in_col': 'monthly_inflow',
    'cash_out_col': 'monthly_outflow'
}

result = preprocess_to_abc_format(df, config)
```

## ⚠️ 주의사항

1. **필수 컬럼**: 현금 유입/유출 컬럼은 반드시 필요합니다.
2. **날짜 형식**: 날짜가 올바른 형식인지 확인하세요.
3. **데이터 타입**: 숫자 데이터는 숫자 또는 문자열 형식이어야 합니다.
4. **결측치**: 날짜가 없는 행은 자동으로 제거됩니다.

## 🐛 에러 처리

### 에러: 컬럼을 찾을 수 없음

```python
ValueError: 날짜 컬럼 'Date'을 찾을 수 없습니다.
```

**해결**: `config`에서 올바른 컬럼명을 지정하세요.

### 에러: 날짜 형식이 맞지 않음

```python
ValueError: time data '31/01/2023' does not match format '%Y-%m-%d'
```

**해결**: `date_format` 파라미터를 올바르게 지정하세요.

## 📚 추가 리소스

- [pandas 문서](https://pandas.pydata.org/docs/)
- [strftime 형식 참조](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)

## 🤝 기여

새로운 데이터 형식을 추가하거나 개선사항이 있으면 언제든 제안해주세요!

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
