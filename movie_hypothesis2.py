import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import linregress
import numpy as np
import re
import matplotlib.pyplot as plt

TMDB_PATH    = "tmdb_5000_movies.csv"
NUMBERS_PATH = "movie_budgets_and_revenues.csv"

MIN_BUDGET  = 1000000   # 제작비 최솟값 (100만 달러 미만은 데이터 오류로 간주)
MIN_REVENUE = 100000    # 수익 최솟값 (10만 달러 미만 제거)
MIN_VOTES   = 10        # 평점 신뢰도를 위한 최소 투표 수

# 맑은 고딕 << 윈도우 환경
def set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'

set_korean_font()

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
############################## 전처리 ##############################

# 가설 2
def pearson_correlation(x, y):
    """
    scipy.stats.pearsonr를 이용한 피어슨 상관분석
    반환: (r, p_value)
    """
    mask = ~(np.isnan(x) | np.isnan(y))  # 둘 중 하나라도 NaN이면 True → 반전으로 유효한 값만 선택
    x, y = x[mask], y[mask]

    r, p_value = pearsonr(x, y)
    return r, p_value


# 회귀선
def linear_regression(x, y):
    """
    scipy.stats.linregress 선형회귀.
    y = beta0 + beta1 * x를 목표

    반환: (beta0, beta1, r_squared)
    """
    mask = ~(np.isnan(x) | np.isnan(y))  # 둘 중 하나라도 NaN이면 제거
    x, y = x[mask], y[mask]

    # linregress: 기울기, 절편, r, p-value, 표준오차 한번에 반환
    beta1, beta0, r, p, se = linregress(x, y)   #  p, se 안 씀
    r_squared = r ** 2  # R^2 = r^2

    return beta0, beta1, r_squared  # beta1이 중요


def analyze_hypothesis2(df):
    """
    가설 2 분석: 평점 <-> 수익 피어슨 상관분석 + 선형회귀.
    """
    print("=" * 60)
    print("[가설 2] 평점 <-> 수익 피어슨 상관분석 + 선형회귀")
    print("=" * 60)

    vote  = df['vote_average'].values
    log_r = df['log_revenue'].values

    # 피어슨 상관분석
    r, p = pearson_correlation(vote, log_r)
    print(f"\n[피어슨 상관분석]")
    print(f"피어슨 r = {r:.4f},  p-value = {p:.6f}")
    print(f"해석: {'유의미' if p < 0.05 else '유의미하지 않은'} "
          f"{'양의' if r > 0 else '음의'} 선형 관계")

    # 선형회귀
    beta0, beta1, r2 = linear_regression(vote, log_r)
    print(f"\n[선형회귀]")
    print(f"회귀식: log_revenue = {beta1:.4f} * vote_average + {beta0:.4f}")
    print(f"R^2 = {r2:.4f}  (모델이 수익 분산의 {r2*100:.1f}%를 설명)")
    print(f"해석: 평점이 1점 오를 때 log_revenue가 {beta1:.4f} 증가")
    print(f"-> 실제 수익은 약 {10**beta1:.2f}배 변화")

    # 결론
    print(f"\n[결론]")
    if p < 0.05 and r > 0.5:   # 0.5 부터 "유의미함"을 의미하게끔 기준을 잡음.
        print(f"통계적으로 유의미한 양의 상관관계 존재 -> 가설 2 지지")
        if r2 < 0.25:
            print(f"R^2={r2:.4f}로 설명력도 떨어짐")
    else:
        print(f"유의미한 상관관계 없음 -> 가설 2 기각")

    print(f"\n참고) n={len(df):,}으로 표본이 크기 때문에 p-value가 매우 작게 나옴")
    print(f"실질적 의미는 r 값과 R^2 값으로 판단할 것")

    return r, r2, beta0, beta1


# 시각화
def plot_scatter(df, r, r2, beta0, beta1):
    """
    산점도 + 회귀선: vote_average vs log_revenue.

    - 산점도: 각 영화를 점으로 표시 (alpha로 겹침 완화)
    - 회귀선: 최소제곱법으로 구한 직선 (추세 파악)
    - x축: vote_average (평점 원본 값)
    - y축: log_revenue (수익 로그 변환값)

    저장: hypothesis2_scatter.png
    """
    vote  = df['vote_average'].values
    log_r = df['log_revenue'].values

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # 산점도: alpha=0.3으로 반투명하게 (점이 많아 겹치므로)
    ax.scatter(vote, log_r, alpha=0.3, color='#4a9e2f', s=20, zorder=3)

    # 회귀선 그리기 (x 범위 전체에 걸쳐 직선)
    x_line = np.linspace(np.nanmin(vote), np.nanmax(vote), 100)
    y_line = beta0 + beta1 * x_line
    ax.plot(x_line, y_line, color='#e24b4a', linewidth=2,
            label=f'회귀선  y = {beta1:.3f}x + {beta0:.3f}  (R^2={r2:.4f})', zorder=4)

    # 제목 및 축 레이블
    ax.set_title(f'평점 vs 수익 (수익 로그 스케일)  r = {r:.4f}',
                 color='white', fontsize=13, pad=14)
    ax.set_xlabel('평균 평점 (vote_average)', color='white', fontsize=11)
    ax.set_ylabel('log10(수익)', color='white', fontsize=11)
    ax.tick_params(colors='white')

    # 축 스타일
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(color='#333', linewidth=0.6, zorder=1)

    # y축 눈금에 실제 수익 금액 표시 (로그 값 -> 달러 환산)
    tick_labels = {5: '$100K', 6: '$1M', 7: '$10M', 8: '$100M', 9: '$1B'}
    yticks = [t for t in tick_labels if np.nanmin(log_r) <= t <= np.nanmax(log_r)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([tick_labels[t] for t in yticks], color='white')

    ax.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig('hypothesis2_scatter.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a1a')
    print("저장: hypothesis2_scatter.png")
    plt.show()


# main
if __name__ == "__main__":

    # 1. 전처리
    df = init(TMDB_PATH, NUMBERS_PATH)
    print(f"\n최종 데이터: {len(df):,}행\n")

    # 2. 분석
    r, r2, beta0, beta1 = analyze_hypothesis2(df)

    # 3. 시각화
    print("\n[산점도 생성 중...]")
    plot_scatter(df, r, r2, beta0, beta1)