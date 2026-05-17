import pandas as pd
import numpy as np
import re

MIN_BUDGET = 1000000
MIN_REVENUE = 100000
MIN_VOTES = 10


def load_data(tmdb_path, numbers_path):
    tmdb_raw = pd.read_csv(tmdb_path, encoding='utf-8')
    numbers_raw = pd.read_csv(numbers_path, encoding='utf-8')
    return tmdb_raw, numbers_raw


def extract_main_country(raw):
    # json 내에 iso_3166_1 값에 국가 코드 있음
    if pd.isna(raw):
        return 'Unknown'
    match = re.search(r'"iso_3166_1":\s*"([A-Z]{2})"', str(raw))
    if match:
        return match.group(1)
    return 'Unknown'    # 나라 데이터 없을 때도 있어서 남겨둠


def clean_currency(value):
    # ex) $54,321,321 -> 54321321.0
    # return floating 변수임에 유의
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(value))  # 숫자와 소수점 제외 전부 제거
    return float(cleaned) if cleaned else np.nan


def preprocess_tmdb(tmdb_raw):
    # 다른건 안씀
    cols_needed = ['title', 'budget', 'revenue', 'vote_average',
                   'vote_count', 'production_countries', 'original_language']
    available_cols = [c for c in cols_needed if c in tmdb_raw.columns]
    tmdb = tmdb_raw[available_cols].copy()

    # 자료형 변환
    tmdb['budget'] = pd.to_numeric(tmdb['budget'])
    tmdb['revenue'] = pd.to_numeric(tmdb['revenue'])
    tmdb['vote_average'] = pd.to_numeric(tmdb['vote_average'])
    tmdb['vote_count'] = pd.to_numeric(tmdb['vote_count'])

    # 결측값 & 이상치 제거
    n_before = len(tmdb)
    tmdb = tmdb[tmdb['budget'] >= MIN_BUDGET]
    tmdb = tmdb[tmdb['revenue'] >= MIN_REVENUE]
    tmdb = tmdb[tmdb['vote_average'].notna() & (tmdb['vote_average'] > 0)]
    tmdb = tmdb[tmdb['vote_count'] >= MIN_VOTES]
    print(f"짤린 데이터 {n_before}/{len(tmdb)} {n_before - len(tmdb)}")

    # json에서 추출한 나라 데이터로 나라 컬럼(main_country) 생성
    tmdb['main_country'] = tmdb['production_countries'].apply(extract_main_country)

    # 제목 정규화 (병합 키)
    tmdb['title_key'] = tmdb['title'].str.lower().str.strip()
    tmdb['title_key'] = tmdb['title_key'].apply(
        lambda x: re.sub(r'[^a-z0-9\s]', '', str(x)).strip()  # 특수문자, 앞뒤 공백 제거, 간단해서 람다로 함
    )

    return tmdb


def preprocess_numbers(numbers_raw):
    numbers = numbers_raw.copy()

    # ex) $54,321,321 -> 54321321.0
    numbers['nb_budget'] = numbers['Budget'].apply(clean_currency)
    numbers['nb_revenue'] = numbers['Worldwide Gross'].apply(clean_currency)

    # 제목 정규화 (병합 키) - TMDB와 동일한 방식으로 (가끔 제목이랑 매칭 안되는 경우 있음)
    numbers['title_key'] = numbers['Movie Name'].str.lower().str.strip()
    numbers['title_key'] = numbers['title_key'].apply(
        lambda x: re.sub(r'[^a-z0-9\s]', '', str(x)).strip()
    )

    # 이상치 제거 - TMDB와 동일한 기준
    n_before = len(numbers)
    numbers = numbers[numbers['nb_budget'] >= MIN_BUDGET]
    numbers = numbers[numbers['nb_revenue'] >= MIN_REVENUE]
    print(f"짤린거: {n_before}/{len(numbers)}")

    return numbers


def merge_datasets(tmdb, numbers):
    # left join, 엑셀로 치면 오른쪽에 The Numbers 데이터 붙인다 보면 됨. 없으면 빈칸
    merged = pd.merge(
        tmdb,
        numbers[['title_key', 'nb_budget', 'nb_revenue']],
        on='title_key',
        how='left'
    )

    matched = merged['nb_budget'].notna().sum()
    print(f"\n전체 {len(merged)}행 중 The Numbers와 매칭된 행: {matched}개")
    print(f"매칭 안 된 행(TMDB 재무 데이터 사용): {len(merged) - matched}개")

    # 재무 정보 The Numbers를 우선 적용하게끔 combine_first()
    merged['final_budget'] = merged['nb_budget'].combine_first(merged['budget'])
    merged['final_revenue'] = merged['nb_revenue'].combine_first(merged['revenue'])

    # 병합 후 최종 필터링
    merged = merged[merged['final_budget'] >= MIN_BUDGET]
    merged = merged[merged['final_revenue'] >= MIN_REVENUE]

    return merged.reset_index(drop=True)


def add_derived_variables(df):
    # 로그 변환 (log10)
    # $1,000,000   -> 6.0
    # $10,000,000  -> 7.0  (10배 차이 -> 로그 스케일에서 1 차이) (지진 진도라고 생각하면 편함)
    # $100,000,000 -> 8.0
    df['log_budget'] = np.log10(df['final_budget'])
    df['log_revenue'] = np.log10(df['final_revenue'])

    # ROI = revenue / budget
    # ROI = 1.0: 손익분기점, ROI = 2.0: 2배 수익, ROI < 1.0: 손실
    df['roi'] = df['final_revenue'] / df['final_budget']

    return df


def init(tmdb_path, numbers_path):
    # 전처리 전체 파이프라인. 분석 파일에서 이것만 호출하면 됨.
    # 반환값: 분석에 바로 쓸 수 있는 df
    tmdb_raw, numbers_raw = load_data(tmdb_path, numbers_path)
    tmdb = preprocess_tmdb(tmdb_raw)
    numbers = preprocess_numbers(numbers_raw)
    df = merge_datasets(tmdb, numbers)
    df = add_derived_variables(df)
    return df

# Test
if __name__ == "__main__":
    df = init("tmdb_5000_movies.csv", "movie_budgets_and_revenues.csv")
    print(df.head())
    print(f"\n최종 행 수: {len(df)}")
