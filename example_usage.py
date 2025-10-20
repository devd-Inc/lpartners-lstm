"""
전처리 함수 사용 예시
다양한 형식의 데이터를 abc_corp_financials 형식으로 변환하는 방법
"""

import pandas as pd
from preprocess_data import preprocess_to_abc_format


# ============================================================================
# 사용 예시 1: 한국어 컬럼명 데이터 변환
# ============================================================================
def example_1_korean_columns():
    print("=" * 80)
    print("예시 1: 한국어 컬럼명 데이터 변환")
    print("=" * 80)
    
    # CSV 파일 읽기
    df = pd.read_csv('example_korean_columns.csv')
    print("\n원본 데이터:")
    print(df.head())
    
    # 컬럼 매핑 설정
    config = {
        'date_col': '날짜',
        'revenue_col': '매출',
        'cash_in_col': '현금유입',
        'cash_out_col': '현금유출'
    }
    
    # 전처리 실행
    result = preprocess_to_abc_format(df, config)
    
    print("\n변환된 데이터:")
    print(result.head())
    
    # CSV로 저장
    result.to_csv('converted_korean.csv', index=False)
    print("\n변환된 데이터가 'converted_korean.csv'로 저장되었습니다.")


# ============================================================================
# 사용 예시 2: 다른 날짜 형식 데이터 변환
# ============================================================================
def example_2_different_date_format():
    print("\n" + "=" * 80)
    print("예시 2: 다른 날짜 형식 (DD/MM/YYYY) 데이터 변환")
    print("=" * 80)
    
    # CSV 파일 읽기
    df = pd.read_csv('example_different_date_format.csv')
    print("\n원본 데이터:")
    print(df.head())
    
    # 컬럼 매핑 및 날짜 형식 설정
    config = {
        'date_col': 'transaction_date',
        'revenue_col': 'sales',
        'cash_in_col': 'inflow',
        'cash_out_col': 'outflow',
        'date_format': '%d/%m/%Y'  # 날짜 형식 지정
    }
    
    # 전처리 실행
    result = preprocess_to_abc_format(df, config)
    
    print("\n변환된 데이터:")
    print(result.head())
    
    # CSV로 저장
    result.to_csv('converted_date_format.csv', index=False)
    print("\n변환된 데이터가 'converted_date_format.csv'로 저장되었습니다.")


# ============================================================================
# 사용 예시 3: 문자열 숫자 데이터 변환
# ============================================================================
def example_3_string_numbers():
    print("\n" + "=" * 80)
    print("예시 3: 문자열 숫자 데이터 변환")
    print("=" * 80)
    
    # CSV 파일 읽기
    df = pd.read_csv('example_string_numbers.csv')
    print("\n원본 데이터:")
    print(df.head())
    print("\n원본 데이터 타입:")
    print(df.dtypes)
    
    # 컬럼 매핑 설정
    config = {
        'date_col': 'date',
        'revenue_col': 'revenue',
        'cash_in_col': 'inflow',
        'cash_out_col': 'outflow'
    }
    
    # 전처리 실행 (자동으로 숫자로 변환)
    result = preprocess_to_abc_format(df, config)
    
    print("\n변환된 데이터:")
    print(result.head())
    print("\n변환된 데이터 타입:")
    print(result.dtypes)
    
    # CSV로 저장
    result.to_csv('converted_string.csv', index=False)
    print("\n변환된 데이터가 'converted_string.csv'로 저장되었습니다.")


# ============================================================================
# 사용 예시 4: 추가 컬럼이 있는 데이터 변환
# ============================================================================
def example_4_extra_columns():
    print("\n" + "=" * 80)
    print("예시 4: 추가 컬럼이 있는 데이터 변환")
    print("=" * 80)
    
    # CSV 파일 읽기
    df = pd.read_csv('example_extra_columns.csv')
    print("\n원본 데이터:")
    print(df.head())
    print(f"\n원본 컬럼: {list(df.columns)}")
    
    # 컬럼 매핑 설정 (기본값 사용)
    config = {
        'date_col': 'Date',
        'revenue_col': 'Revenue',
        'cash_in_col': 'Cash_Inflow',
        'cash_out_col': 'Cash_Outflow'
    }
    
    # 전처리 실행 (필요한 컬럼만 추출)
    result = preprocess_to_abc_format(df, config)
    
    print("\n변환된 데이터:")
    print(result.head())
    print(f"\n변환된 컬럼: {list(result.columns)}")
    
    # CSV로 저장
    result.to_csv('converted_extra.csv', index=False)
    print("\n변환된 데이터가 'converted_extra.csv'로 저장되었습니다.")


# ============================================================================
# 사용 예시 5: 사용자 정의 데이터 변환
# ============================================================================
def example_5_custom_data():
    print("\n" + "=" * 80)
    print("예시 5: 사용자 정의 데이터 변환")
    print("=" * 80)
    
    # 사용자가 가진 데이터 (예시)
    custom_data = pd.DataFrame({
        '기간': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05'],
        '총매출': [1000, 1100, 1200, 1300, 1400],
        '입금액': [800, 900, 1000, 1100, 1200],
        '출금액': [600, 700, 800, 900, 1000]
    })
    
    print("\n사용자 데이터:")
    print(custom_data)
    
    # 컬럼 매핑 설정
    config = {
        'date_col': '기간',
        'revenue_col': '총매출',
        'cash_in_col': '입금액',
        'cash_out_col': '출금액',
        'date_format': '%Y-%m'  # 날짜 형식 지정
    }
    
    # 전처리 실행
    result = preprocess_to_abc_format(custom_data, config)
    
    print("\n변환된 데이터:")
    print(result)
    
    # CSV로 저장
    result.to_csv('converted_custom.csv', index=False)
    print("\n변환된 데이터가 'converted_custom.csv'로 저장되었습니다.")


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("전처리 함수 사용 예시")
    print("=" * 80)
    
    # 각 예시 실행
    example_1_korean_columns()
    example_2_different_date_format()
    example_3_string_numbers()
    example_4_extra_columns()
    example_5_custom_data()
    
    print("\n" + "=" * 80)
    print("모든 예시가 완료되었습니다!")
    print("=" * 80)

