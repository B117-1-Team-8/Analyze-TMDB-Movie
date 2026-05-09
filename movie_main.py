# movie_main.py
import pandas as pd
import numpy as np
import re

TMDB_PATH = "tmdb_5000_movies.csv"           # TMDB 5000 영화 데이터
NUMBERS_PATH = "movie_budgets_and_revenues.csv"  # The Numbers 재무 데이터

MIN_BUDGET = 1000000
MIN_REVENUE = 100000
MIN_VOTES  = 10

tmdb_raw = pd.read_csv(TMDB_PATH, encoding='utf-8')
numbers_raw = pd.read_csv(NUMBERS_PATH, encoding='utf-8')

#print("TMDB 전처리")

# 다른건 안씀
cols_needed = ['title', 'budget', 'revenue', 'vote_average',
               'vote_count', 'production_countries', 'original_language']
# cols_needed에 있는 열이 진짜 있는지 확인
available_cols = [c for c in cols_needed if c in tmdb_raw.columns]
# available_cols 리스트 내 열 이름으로 실제 쓸 데이터(열) 추출
tmdb = tmdb_raw[available_cols].copy()

# 자료형 변환
tmdb['budget'] = pd.to_numeric(tmdb['budget'])
tmdb['revenue'] = pd.to_numeric(tmdb['revenue'])
tmdb['vote_average'] = pd.to_numeric(tmdb['vote_average'])
tmdb['vote_count'] = pd.to_numeric(tmdb['vote_count'])

# 결측값 & 이상치 제거
n_before = len(tmdb)
tmdb = tmdb[tmdb['budget'] >= MIN_BUDGET]   # 예산 100만 달러 밑은 컷
tmdb = tmdb[tmdb['revenue'] >= MIN_REVENUE] # 수입 10만 달러 밑은 컷
tmdb = tmdb[tmdb['vote_average'].notna() & (tmdb['vote_average'] > 0)] # 평점 na인 경우, 0인 경우 컷
tmdb = tmdb[tmdb['vote_count'] >= MIN_VOTES] # 너무 저조한 투표수 컷
n_after = len(tmdb)

# 짤린 데이터 4803/3055 1748   # 재무 정보가 없는 경우가 대다수...
# print(f"짤린 데이터 {n_before}/{n_after} {n_before - n_after}")

# production_countries 파싱
# 국가 코드 별로 갖고오기 json 내에 iso_3166_1 값에 국가 코드 있음.
def extract_main_country(raw):
    if pd.isna(raw):
        return 'Unknown'
    match = re.search(r'"iso_3166_1":\s*"([A-Z]{2})"', str(raw))
    if match:
        return match.group(1)
    return 'Unknown'    # 나라 데이터 필요 없을 때도 있어서 남겨둠

# json에서 추출한 나라 데이터로 나라 컬럼(main_country) 생성
tmdb['main_country'] = tmdb['production_countries'].apply(extract_main_country)

#---------------------------------------------------------------------------------------

# numbers data 전처리
numbers = numbers_raw.copy()

budget_col  = 'Budget'
revenue_col = 'Worldwide Gross'
title_col   = 'Movie Name'

# ex) $54,321,321 -> 54321321.0
# return floating 변수임에 유의
def clean_currency(value):
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(value)) # 숫자와 소수점 제외 전부 제거
    return float(cleaned) if cleaned else np.nan

# 나중에 TMDB랑 병합할 때도 쓸 거임
numbers['nb_budget']  = numbers['Budget'].apply(clean_currency)
numbers['nb_revenue'] = numbers['Worldwide Gross'].apply(clean_currency)
numbers['title_key']  = numbers['Movie Name'].str.lower().str.strip()

# 이상치 제거 # TMDB와 동일한 기준
n_before = len(numbers)
numbers = numbers[numbers['nb_budget']  >= MIN_BUDGET]
numbers = numbers[numbers['nb_revenue'] >= MIN_REVENUE]

#print(f"짤린거: {n_before}/{len(numbers)}")


# 두 데이터셋 병합
#
#   [병합 키: 영화 제목]
#   두 데이터셋에 공통된 고유 ID가 없으므로 영화 제목을 병합 기준으로 사용
#   단, 제목이 대소문자, 특수문자, 공백 차이로 일치하지 않을 수 있음 << 일일이 확인은 .. 힘듦
#   ex) TMDB "The Dark Knight" vs Numbers "dark knight the"
#
#   [Left Join 선택 이유]
#   TMDB가 메인 데이터 (5000개 영화의 메타데이터 보유)
#   The Numbers는 재무 정보만 추가하는 보조 역할
#   → TMDB 기준 left join: TMDB에 있는 모든 행 유지,
#     Numbers에서 매칭되면 재무 정보 추가, 매칭 안 되면 NaN
#
#   [combine_first() - 데이터 우선순위]
#   The Numbers 재무 정보가 있으면 우선 사용 (신뢰도 높음)
#   없으면(NaN이면) TMDB 재무 정보로 대체
#   series_a.combine_first(series_b): a가 NaN인 위치를 b 값으로 채움

# TMDB 제목 정규화
tmdb['title_key'] = tmdb['title'].str.lower().str.strip()   # 소문자로 싹다 변환, 앞뒤 공백 제거
tmdb['title_key'] = tmdb['title_key'].apply(
    lambda x: re.sub(r'[^a-z0-9\s]', '', str(x)).strip()    # 특수문자, 앞뒤 공백 제거, 간단해서 람다로 함
)

# left join
if 'title_key' in numbers.columns:
    merged = pd.merge(
        tmdb,
        numbers[['title_key', 'nb_budget', 'nb_revenue']],
        on='title_key',
        how='left'  # left join, 액셀로 치면 오른쪽에 The Numbers 데이터 붙인다 보면 됨. 없으면 빈칸
    )
    matched = merged['nb_budget'].notna().sum()
    #print(f"\n전체 {len(merged)}행 중 The Numbers와 매칭된 행: {matched}개")
    #print(f"매칭 안 된 행(TMDB 재무 데이터 사용해야하는 거): {len(merged) - matched}개")

    # 재무 정보 The Numbers를 우선 적용하게끔 combine_first()
    merged['final_budget'] = merged['nb_budget'].combine_first(merged['budget'])
    merged['final_revenue'] = merged['nb_revenue'].combine_first(merged['revenue'])
else:
    merged = tmdb.copy()
    merged['final_budget'] = merged['budget']
    merged['final_revenue'] = merged['revenue']

# 병합 후 최종 필터링
merged = merged[merged['final_budget']  >= MIN_BUDGET]
merged = merged[merged['final_revenue'] >= MIN_REVENUE]
df = merged.reset_index(drop=True)

#
#   [로그 변환 (log10)이 필요한 이유]
#   영화 제작비/수익의 실제 분포:
#   저예산 독립영화: 수백만 달러
#   대형 영화     : 수십억 달러
#   → 분포가 오른쪽으로 매우 치우침
#
#   예시:
#   $1,000,000   -> log10 = 6.0
#   $10,000,000  -> log10 = 7.0   (10배 차이 -> 로그 스케일에서 1 차이) (지진 진도라고 생각하면 편함)
#   $100,000,000 -> log10 = 8.0
#   $1,000,000,000 -> log10 = 9.0
#
#   분포가 정규분포에 가까워짐 -> 상관/회귀분석 가정 충족
#   이상치(초대형 흥행작)의 영향이 줄어듦
#   상관계수 r이 더 정확하게 계산됨
#
#   [ROI 해석할 때]
#   ROI = revenue / budget
#   ROI = 1.0: 제작비만 회수 (손익분기점)
#   ROI = 2.0: 제작비의 2배 수익
#   ROI < 1.0: 손실
#   가설 1의 보조 지표로 활용 가능할지도?

df['log_budget'] = np.log10(df['final_budget'])            # 로그 변환
df['log_revenue'] = np.log10(df['final_revenue'])           # 로그 변환
df['roi'] = df['final_revenue'] / df['final_budget']    # ROI

#print(f"\n  [로그 변환 예시]")
#sample = df[['title', 'final_budget', 'log_budget',
#             'final_revenue', 'log_revenue', 'roi']].head(3)
#print(sample.to_string(index=False))

#print(f"\n  [분포 비교: 변환 전 vs 후]")
#print(f"  final_budget  - 평균: ${df['final_budget'].mean():>15,.0f}  "
#      f"표준편차: ${df['final_budget'].std():>15,.0f}")
#print(f"  log_budget    - 평균: {df['log_budget'].mean():>6.4f}           "
#      f"표준편차: {df['log_budget'].std():>6.4f}  ← 훨씬 안정적")

print(f"""
최종 데이터 컬럼:
- title          : 영화 제목
- final_budget   : 최종 제작비 (The Numbers 우선, 없으면 TMDB)    [전처리에서 적용된 사항]
- final_revenue  : 최종 수익  (The Numbers 우선, 없으면 TMDB)     [전처리에서 적용된 사항]
- vote_average   : 평균 평점
- vote_count     : 투표 수
- main_country   : 주 제작 국가 (ISO 2자리 코드)
- original_language : 원본 언어
- log_budget     : 제작비 log10 변환값  [신규]
- log_revenue    : 수익 log10 변환값    [신규]
- roi            : 투자 수익률          [신규]
""")

print("이후 이 df를 가설 1, 2, 3 분석 함수에 그대로 전달.")
# main()에 init processing 과정에 바로 적용하면 됨.
