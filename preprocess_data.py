import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_to_abc_format(df, config=None):
    """
    다양한 형식의 입력 데이터를 abc_corp_financials 형식으로 변환
    
    Parameters:
    -----------
    df : pandas.DataFrame
        입력 데이터프레임
    config : dict, optional
        변환 설정 딕셔너리
        - date_col: 날짜 컬럼명
        - revenue_col: 매출 컬럼명
        - cash_in_col: 현금유입 컬럼명
        - cash_out_col: 현금유출 컬럼명
        - date_format: 날짜 형식 (strftime format)
    
    Returns:
    --------
    pandas.DataFrame
        변환된 데이터프레임 (abc_corp_financials 형식)
    """
    
    # 기본 설정
    default_config = {
        'date_col': 'Date',
        'revenue_col': 'Revenue',
        'cash_in_col': 'Cash_Inflow',
        'cash_out_col': 'Cash_Outflow',
        'date_format': '%Y-%m-%d'
    }
    
    if config:
        default_config.update(config)
    
    config = default_config
    
    # 결과 데이터프레임 초기화
    result_df = pd.DataFrame()
    
    # 1. 날짜 변환
    if config['date_col'] in df.columns:
        result_df['Date'] = pd.to_datetime(df[config['date_col']], format=config['date_format'], errors='coerce')
    else:
        raise ValueError(f"날짜 컬럼 '{config['date_col']}'을 찾을 수 없습니다.")
    
    # 2. 매출 변환
    if config['revenue_col'] in df.columns:
        result_df['Revenue'] = pd.to_numeric(df[config['revenue_col']], errors='coerce')
    else:
        print(f"경고: 매출 컬럼 '{config['revenue_col']}'을 찾을 수 없습니다. NaN으로 채웁니다.")
        result_df['Revenue'] = np.nan
    
    # 3. 현금유입 변환
    if config['cash_in_col'] in df.columns:
        result_df['Cash_Inflow'] = pd.to_numeric(df[config['cash_in_col']], errors='coerce')
    else:
        raise ValueError(f"현금유입 컬럼 '{config['cash_in_col']}'을 찾을 수 없습니다.")
    
    # 4. 현금유출 변환
    if config['cash_out_col'] in df.columns:
        result_df['Cash_Outflow'] = pd.to_numeric(df[config['cash_out_col']], errors='coerce')
    else:
        raise ValueError(f"현금유출 컬럼 '{config['cash_out_col']}'을 찾을 수 없습니다.")
    
    # 5. 순 현금 흐름 계산
    result_df['Cash_Flow'] = result_df['Cash_Inflow'] - result_df['Cash_Outflow']
    
    # 6. 결측치 처리 (선택적)
    # 날짜가 없는 행 제거
    result_df = result_df.dropna(subset=['Date'])
    
    # 7. 정렬
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    
    return result_df


# ============================================================================
# 예시 1: 한국어 컬럼명
# ============================================================================
def create_example_korean_columns():
    """한국어 컬럼명을 가진 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    df = pd.DataFrame({
        '날짜': dates,
        '매출': np.cumsum(np.random.normal(100, 20, 12)) + 1000,
        '현금유입': np.random.uniform(800, 1200, 12),
        '현금유출': np.random.uniform(700, 1000, 12)
    })
    
    return df


# ============================================================================
# 예시 2: 다른 날짜 형식 (DD/MM/YYYY)
# ============================================================================
def create_example_different_date_format():
    """다른 날짜 형식을 가진 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    dates_str = [d.strftime('%d/%m/%Y') for d in dates]
    
    df = pd.DataFrame({
        'transaction_date': dates_str,
        'sales': np.cumsum(np.random.normal(100, 20, 12)) + 1000,
        'inflow': np.random.uniform(800, 1200, 12),
        'outflow': np.random.uniform(700, 1000, 12)
    })
    
    return df


# ============================================================================
# 예시 3: 누적값 대신 월별 순증가값
# ============================================================================
def create_example_monthly_values():
    """월별 순증가값을 가진 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    df = pd.DataFrame({
        'period': dates,
        'monthly_revenue': np.random.normal(100, 20, 12),  # 누적이 아닌 월별값
        'monthly_inflow': np.random.uniform(800, 1200, 12),
        'monthly_outflow': np.random.uniform(700, 1000, 12)
    })
    
    return df


# ============================================================================
# 예시 4: 다른 데이터 타입 (문자열 숫자)
# ============================================================================
def create_example_string_numbers():
    """문자열로 된 숫자 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    df = pd.DataFrame({
        'date': dates,
        'revenue': [f"{x:.2f}" for x in np.cumsum(np.random.normal(100, 20, 12)) + 1000],
        'inflow': [f"{x:.2f}" for x in np.random.uniform(800, 1200, 12)],
        'outflow': [f"{x:.2f}" for x in np.random.uniform(700, 1000, 12)]
    })
    
    return df


# ============================================================================
# 예시 5: 추가 컬럼이 있는 데이터
# ============================================================================
def create_example_extra_columns():
    """추가 컬럼이 있는 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    df = pd.DataFrame({
        'Date': dates,
        'Revenue': np.cumsum(np.random.normal(100, 20, 12)) + 1000,
        'Cash_Inflow': np.random.uniform(800, 1200, 12),
        'Cash_Outflow': np.random.uniform(700, 1000, 12),
        'Company_ID': ['ABC_CORP'] * 12,  # 추가 컬럼
        'Region': ['Seoul'] * 12,  # 추가 컬럼
        'Notes': ['Sample data'] * 12  # 추가 컬럼
    })
    
    return df


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("데이터 전처리 예시")
    print("=" * 80)
    
    # 예시 1: 한국어 컬럼명
    print("\n[예시 1] 한국어 컬럼명 데이터")
    print("-" * 80)
    df_korean = create_example_korean_columns()
    print("원본 데이터:")
    print(df_korean.head())
    
    config_korean = {
        'date_col': '날짜',
        'revenue_col': '매출',
        'cash_in_col': '현금유입',
        'cash_out_col': '현금유출'
    }
    
    result_korean = preprocess_to_abc_format(df_korean, config_korean)
    print("\n변환된 데이터:")
    print(result_korean.head())
    
    # 예시 2: 다른 날짜 형식
    print("\n\n[예시 2] 다른 날짜 형식 (DD/MM/YYYY)")
    print("-" * 80)
    df_date_format = create_example_different_date_format()
    print("원본 데이터:")
    print(df_date_format.head())
    
    config_date = {
        'date_col': 'transaction_date',
        'revenue_col': 'sales',
        'cash_in_col': 'inflow',
        'cash_out_col': 'outflow',
        'date_format': '%d/%m/%Y'
    }
    
    result_date = preprocess_to_abc_format(df_date_format, config_date)
    print("\n변환된 데이터:")
    print(result_date.head())
    
    # 예시 3: 월별 순증가값
    print("\n\n[예시 3] 월별 순증가값 (누적 변환 필요)")
    print("-" * 80)
    df_monthly = create_example_monthly_values()
    print("원본 데이터:")
    print(df_monthly.head())
    
    config_monthly = {
        'date_col': 'period',
        'revenue_col': 'monthly_revenue',
        'cash_in_col': 'monthly_inflow',
        'cash_out_col': 'monthly_outflow'
    }
    
    # 누적값으로 변환
    df_monthly_cumsum = df_monthly.copy()
    df_monthly_cumsum['monthly_revenue'] = df_monthly['monthly_revenue'].cumsum()
    
    result_monthly = preprocess_to_abc_format(df_monthly_cumsum, config_monthly)
    print("\n변환된 데이터 (누적값):")
    print(result_monthly.head())
    
    # 예시 4: 문자열 숫자
    print("\n\n[예시 4] 문자열 숫자 데이터")
    print("-" * 80)
    df_string = create_example_string_numbers()
    print("원본 데이터:")
    print(df_string.head())
    print("\n원본 데이터 타입:")
    print(df_string.dtypes)
    
    config_string = {
        'date_col': 'date',
        'revenue_col': 'revenue',
        'cash_in_col': 'inflow',
        'cash_out_col': 'outflow'
    }
    
    result_string = preprocess_to_abc_format(df_string, config_string)
    print("\n변환된 데이터:")
    print(result_string.head())
    print("\n변환된 데이터 타입:")
    print(result_string.dtypes)
    
    # 예시 5: 추가 컬럼
    print("\n\n[예시 5] 추가 컬럼이 있는 데이터")
    print("-" * 80)
    df_extra = create_example_extra_columns()
    print("원본 데이터:")
    print(df_extra.head())
    
    result_extra = preprocess_to_abc_format(df_extra)
    print("\n변환된 데이터 (추가 컬럼 제외):")
    print(result_extra.head())
    
    # CSV 파일로 저장
    print("\n\n" + "=" * 80)
    print("예시 데이터를 CSV 파일로 저장합니다...")
    print("=" * 80)
    
    df_korean.to_csv('example_korean_columns.csv', index=False)
    df_date_format.to_csv('example_different_date_format.csv', index=False)
    df_monthly.to_csv('example_monthly_values.csv', index=False)
    df_string.to_csv('example_string_numbers.csv', index=False)
    df_extra.to_csv('example_extra_columns.csv', index=False)
    
    print("\n저장 완료:")
    print("  - example_korean_columns.csv")
    print("  - example_different_date_format.csv")
    print("  - example_monthly_values.csv")
    print("  - example_string_numbers.csv")
    print("  - example_extra_columns.csv")
    
    print("\n" + "=" * 80)
    print("전처리 완료!")
    print("=" * 80)

