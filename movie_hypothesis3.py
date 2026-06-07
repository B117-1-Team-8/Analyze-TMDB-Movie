
import math
import json
import urllib.request
import webbrowser
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import folium

TMDB_V11_PATH = "TMDB_movie_dataset_v11.csv"

MIN_VOTES    = 100   # 평점 신뢰도를 위한 최소 투표 수
MIN_MOVIES   = 50    # 국가 당 신뢰구간 계산에 필요한 최소 영화 수
N_COUNTRIES  = 20    # 분석할 상위 국가 수 (데이터 수 기준)
Z_95         = 1.96  # 신뢰구간 z값 (현재는 95% 기준)

# 국가명 -> ISO 2자리 코드 변환 딕셔너리 (folium 지도 매핑용 - 필요한 만큼 추가 가능)
# 해당 딕셔너리는 국가 데이터가 어떻게 나오는지 확인 후 LLM 사용하여 제작.
COUNTRY_TO_ISO = {
    'United States of America': 'US',
    'United Kingdom':           'GB',
    'France':                   'FR',
    'Germany':                  'DE',
    'Japan':                    'JP',
    'South Korea':              'KR',
    'China':                    'CN',
    'India':                    'IN',
    'Australia':                'AU',
    'Italy':                    'IT',
    'Spain':                    'ES',
    'Canada':                   'CA',
    'Russia':                   'RU',
    'Brazil':                   'BR',
    'Mexico':                   'MX',
    'Sweden':                   'SE',
    'Denmark':                  'DK',
    'Norway':                   'NO',
    'Netherlands':              'NL',
    'Poland':                   'PL',
    'Argentina':                'AR',
    'Turkey':                   'TR',
    'Iran':                     'IR',
    'Hong Kong':                'HK',
    'Belgium':                  'BE',
    'Austria':                  'AT',
    'Switzerland':              'CH',
    'New Zealand':              'NZ',
    'Portugal':                 'PT',
    'Czech Republic':           'CZ',
    'Hungary':                  'HU',
    'Romania':                  'RO',
    'Finland':                  'FI',
    'Greece':                   'GR',
    'Ireland':                  'IE',
    'Thailand':                 'TH',
    'Indonesia':                'ID',
    'Philippines':              'PH',
    'South Africa':             'ZA',
    'Egypt':                    'EG',
    'Israel':                   'IL',
    'Pakistan':                 'PK',
    'Nigeria':                  'NG',
    'Ukraine':                  'UA',
    'Taiwan':                   'TW',
    'Singapore':                'SG',
}


# 한글 폰트 설정
# 한 줄이지만 일단 모듈로 제작
def set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'

set_korean_font()


# 전처리
def extract_main_country(raw):
    # 첫 번째 국가가 제작 국가로 간주됨. (공식)
    if pd.isna(raw) or str(raw).strip() == '':
        return 'Unknown'    # NA 또는 데이터 없으면 Unknown으로 처리
    return str(raw).split(',')[0].strip()   # ','로 split 후 양 끝 스페이스 제거


def load_and_preprocess():
    # status = 'Released': 실제 개봉된 영화만
    # adult = False: 성인물 제외
    # vote_count >= MIN_VOTES: 평점 신뢰도 확보 (현재는 100을 기준)
    # vote_average > 0: 평점 없는 행 제거
    # production_countries 결측 제거
    print("TMDB v11 데이터 로드 및 전처리")

    print("\n데이터 로드 중 (파일이 크므로 시간이 걸릴 수 있음)")
    df = pd.read_csv(
        TMDB_V11_PATH,
        usecols=['title', 'vote_average', 'vote_count', 'status',
                 'adult', 'original_language', 'production_countries'] #쓰는 컬럼만 갖고 오도록 함
    )
    print(f"원본 로드 완료: {len(df):,}편")

    # 필터링
    n_before = len(df)
    df = df[df['status'] == 'Released']                         # 개봉 영화만
    df = df[df['adult'] == False]                               # 성인물 제외
    df['vote_count']   = pd.to_numeric(df['vote_count'],   errors='coerce')  # 숫자 데이터로 바꿈
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')  # errors='coerce' -> 숫자로 안바뀌면 NaN으로 변경하게함
    df = df[df['vote_count']   >= MIN_VOTES]                    # 최소 투표 수 (자체적으로 100표 기준으로 정함)
    df = df[df['vote_average'] >  0]                            # 평점 있는 것만
    df = df[df['production_countries'].notna()]                 # 국가 정보 있는 것만
    df = df[df['production_countries'].str.strip() != '']       # strip 결과가 ''이면 없앰

    print(f"필터링 후: {len(df):,}편  ({n_before - len(df):,}편 제거)")

    # 주 제작 국가 추출 (국가명)
    df['main_country'] = df['production_countries'].apply(extract_main_country) # 위에 함수 참고
    df = df[df['main_country'] != 'Unknown'] # Unknown은 필요 없으니 제거

    # ISO 코드 변환 (지도용)
    df['iso_code'] = df['main_country'].map(COUNTRY_TO_ISO)

    print(f"포함된 국가 수: {df['main_country'].nunique()}개국")
    print(f"\n상위 20개국 영화 수:")  # 20개 기준으로 할거라서 우선 20개로 출력하게끔
    print(df['main_country'].value_counts().head(20).to_string())

    return df.reset_index(drop=True)

# 가설 3 분석

def confidence_interval(arr, z=Z_95):
    # 단일 집단의 평균에 대한 95% 신뢰구간 계산.
    arr = arr[~np.isnan(arr)]
    n = len(arr) # 표본 수
    if n < 30: # 30 미만은 그 국가에는 데이터가 없다고 침
        return np.nan, np.nan, np.nan, np.nan, n

    # 공식에 맞게 각각 구하고 return
    mean   = np.mean(arr)
    std    = np.std(arr, ddof=1)
    se     = std / math.sqrt(n)
    margin = z * se
    return mean, mean - margin, mean + margin, se, n


def check_overlap(ci1, ci2):
    # 두 신뢰구간 (lower, upper) 이 겹치는지 확인.
    # 겹치지 않는 조건: upper1 < lower2 또는 upper2 < lower1
    lower1, upper1 = ci1
    lower2, upper2 = ci2
    return not (upper1 < lower2 or upper2 < lower1)


def compute_country_ci(df):
    # MIN_MOVIES 이상인 국가만 포함
    # 데이터 수 기준 상위 N_COUNTRIES개국 선택
    # 최종 반환은 평균 평점 내림차순 정렬
    results = {}
    for country, group in df.groupby('main_country'):
        arr = group['vote_average'].dropna().values
        if len(arr) < MIN_MOVIES:
            continue
        mean, lower, upper, se, n = confidence_interval(arr)
        iso = COUNTRY_TO_ISO.get(country, country)  # country 국가 데이터에 매칭되는 ISO 값을 다시 country에 저장
        results[iso] = {
            'country_name': country,
            'n':      n,
            'mean':   mean,
            'lower':  lower,
            'upper':  upper,
            'margin': Z_95 * se
        }

    ci_df = pd.DataFrame(results).T # results 딕셔너리는 행에 컬럼이 위치해있어서 전치
    ci_df = ci_df.sort_values('n', ascending=False).head(N_COUNTRIES) # 상위 그룹 20개 구하기 위한 정렬
    ci_df = ci_df.sort_values('mean', ascending=False) # 각 상위 그룹 20개 평균 평점이 높은 순으로 다시 정렬
    return ci_df


def analyze_hypothesis3(df):
    print("가설 3 분석: 제작 국가별 평점 신뢰구간 비교")

    ci_df = compute_country_ci(df)

    #     ISO  국가명        n    평균    하한     상한     마진
    # ex) KR   South Korea  233  7.1985  7.1118  7.2852  0.0867
    # 마진 값은 오차범위로 표본 수가 많으면 보통 낮아짐.
    print(f"\n분석 대상: {len(ci_df)}개국 (영화 수 상위 {N_COUNTRIES}개국, 최소 {MIN_MOVIES}편)")
    # 프린트용
    print(f"\n{'ISO':>5} {'국가명':<30} {'n':>7} {'평균':>7} {'하한':>8} {'상한':>8} {'±':>7}")
    print("  " + "-" * 100)
    for iso, row in ci_df.iterrows():
        print(f"  {iso:>5} {row['country_name']:<30} {int(row['n']):>7,} "
              f"{row['mean']:>7.4f} {row['lower']:>8.4f} {row['upper']:>8.4f} {row['margin']:>7.4f}")
    ###

    # 1위 vs 꼴찌
    top_c = ci_df.index[0]
    bot_c = ci_df.index[-1]
    top_ci = (ci_df.loc[top_c, 'lower'], ci_df.loc[top_c, 'upper'])
    bot_ci = (ci_df.loc[bot_c, 'lower'], ci_df.loc[bot_c, 'upper'])
    overlap_1v1 = check_overlap(top_ci, bot_ci)

    print(f"\n[1위 vs 꼴찌 신뢰구간 비교]")
    print(f"최고: {top_c} ({ci_df.loc[top_c,'country_name']})  평균={ci_df.loc[top_c,'mean']:.4f},  CI=[{top_ci[0]:.4f}, {top_ci[1]:.4f}]")
    print(f"최저: {bot_c} ({ci_df.loc[bot_c,'country_name']})  평균={ci_df.loc[bot_c,'mean']:.4f},  CI=[{bot_ci[0]:.4f}, {bot_ci[1]:.4f}]")
    print(f"겹침 여부: {'겹침 ' if overlap_1v1 else '겹치지 않음 '}")

    # 상위 5 vs 하위 5
    top5_arr = np.concatenate([
        df[df['main_country'] == ci_df.loc[c, 'country_name']]['vote_average'].dropna().values
        for c in ci_df.index[:5]
    ])
    bot5_arr = np.concatenate([
        df[df['main_country'] == ci_df.loc[c, 'country_name']]['vote_average'].dropna().values
        for c in ci_df.index[-5:]
    ])
    mean_t, lower_t, upper_t, _, nt = confidence_interval(top5_arr)
    mean_b, lower_b, upper_b, _, nb = confidence_interval(bot5_arr)
    overlap_group = check_overlap((lower_t, upper_t), (lower_b, upper_b))

    print(f"\n[상위 5개국 vs 하위 5개국 그룹 비교]")
    print(f"상위: {', '.join(ci_df.index[:5])}  n={nt:,}, 평균={mean_t:.4f}, CI=[{lower_t:.4f}, {upper_t:.4f}]")
    print(f"하위: {', '.join(ci_df.index[-5:])}  n={nb:,}, 평균={mean_b:.4f}, CI=[{lower_b:.4f}, {upper_b:.4f}]")
    print(f"그룹 간 겹침 여부: {'겹침' if overlap_group else '겹치지 않음'}")

    # "결론 강화"로 나옴 -> 가설 3을 지지 + 강화
    print(f"\n[결론]")
    if not overlap_1v1:
        print("최고/최저 평점 국가의 95% 신뢰구간이 겹치지 않음")
        print("-> 제작 국가별 영화 평점에 유의미한 차이가 있다. 가설 3 지지")
    else:
        print("최고/최저 평점 국가의 95% 신뢰구간이 겹침")
        print("-> 제작 국가별 영화 평점에 유의미한 차이가 없다. 가설 3 기각")
    if not overlap_group:
        print("상위/하위 5개국 그룹 간 신뢰구간도 겹치지 않음 -> 결론 강화")

    return ci_df


# 시각화
def plot_errorbar(ci_df, grand_mean):
    # ci_df에서 그래프에 쓸 값 추출
    countries = list(ci_df.index)           # x축 레이블 (ISO 코드)
    means     = ci_df['mean'].values        # 각 국가 평균 평점
    margins   = ci_df['margin'].values      # 각 국가 95% CI 오차 범위 (±값)
    ns        = ci_df['n'].values.astype(int)  # 각 국가 데이터 수
    x         = np.arange(len(countries))  # 막대 위치 (0, 1, 2, ...)

    fig, ax = plt.subplots(figsize=(14, 6))

    # 다크 배경
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # 전체 평균보다 높으면 진한 초록, 낮으면 연한 초록으로 색상 구분
    colors = ['#4a9e2f' if m >= grand_mean else '#2a6e1a' for m in means]
    ax.bar(x, means, color=colors, alpha=0.75, width=0.55,
           edgecolor='#6abf50', linewidth=0.8, zorder=3)

    # y축 범위: 최솟값에서 0.5 아래 ~ 최댓값에서 0.6 위
    # 수치 레이블이 막대 위에 표시되므로 위쪽 여유 필요
    ax.set_ylim(max(0, means.min() - 0.5), means.max() + 0.6)

    cap = 0.18  # 에러바 상하단 가로선(캡) 너비
    for i, (m, mg, n_val) in enumerate(zip(means, margins, ns)):
        # 에러바 세로선: 평균 ± margin 구간
        ax.plot([x[i], x[i]], [m-mg, m+mg], color='#cccccc', lw=1.5, zorder=4)
        # 에러바 상단 캡 (가로선)
        ax.plot([x[i]-cap, x[i]+cap], [m+mg, m+mg], color='#cccccc', lw=1.5, zorder=4)
        # 에러바 하단 캡 (가로선)
        ax.plot([x[i]-cap, x[i]+cap], [m-mg, m-mg], color='#cccccc', lw=1.5, zorder=4)

        # 막대 위에 평균 수치 표시 (에러바 상단보다 살짝 위)
        ax.text(x[i], m+mg+0.07, f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, color='white', fontweight='bold')

        # 평균 수치 위에 n= 데이터 수 표시
        ax.text(x[i], m+mg+0.22, f'n={n_val:,}', ha='center', va='bottom',
                fontsize=7.5, color='#aaaaaa')

        # 에러바 오른쪽에 ±margin 수치 표시
        ax.text(x[i]+cap+0.05, m, f'±{mg:.2f}', ha='left', va='center',
                fontsize=7, color='#999999')

    # 전체 평균 점선 (빨간 점선)
    ax.axhline(grand_mean, color='#e24b4a', linestyle='--', linewidth=1.5,
               label=f'전체 평균 {grand_mean:.2f}', zorder=2)

    # 축 설정
    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=10, color='white')
    ax.set_ylabel('평균 평점', color='white', fontsize=11)
    ax.set_xlabel('제작 국가 (ISO 코드)', color='white', fontsize=11)
    ax.set_title('국가별 평균 평점 및 95% 신뢰구간 (TMDB v11)', color='white', fontsize=13, pad=14)
    ax.tick_params(colors='white')

    # 위쪽, 오른쪽 테두리 제거 (깔끔하게)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

    # y축 보조선
    ax.grid(axis='y', color='#333', linewidth=0.6, zorder=1)
    ax.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig('tmdbv11_hypothesis3_errorbar.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a1a')
    print("저장: tmdbv11_hypothesis3_errorbar.png")
    plt.show()

# GeoJSON 및 CSS는 LLM, google css 참고
def plot_folium_map(ci_df):
    # 세계 국가 경계 GeoJSON 로드 (인터넷 연결 필요)
    # 이 GeoJSON은 각 국가의 폴리곤(경계선) 좌표 + ISO 코드 등 속성 포함
    GEOJSON_URL = (
        "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    )
    print("GeoJSON 로드 중...")
    try:
        with urllib.request.urlopen(GEOJSON_URL, timeout=10) as resp:
            geo_data = json.loads(resp.read().decode('utf-8'))
        print("GeoJSON 로드 완료")
    except Exception as e:
        print(f"GeoJSON 로드 실패: {e}")
        return

    # folium Choropleth에 넘길 DataFrame 준비
    # ci_df의 인덱스(ISO 코드)를 iso_a2 컬럼으로 변환
    map_df = ci_df.reset_index().rename(columns={'index': 'iso_a2'})
    map_df['iso_a2'] = map_df['iso_a2'].astype(str)
    map_df['n']      = map_df['n'].astype(int)

    # 다크 배경 타일 지도 생성 (CartoDB dark_matter)
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB dark_matter')

    # Choropleth 레이어: 국가별 평균 평점에 따라 색상 채우기
    # key_on: GeoJSON의 어떤 속성값으로 map_df의 iso_a2와 매칭할지 지정
    # fill_color='YlGn': 낮은 평점 = 연노랑, 높은 평점 = 진초록
    choropleth = folium.Choropleth(
        geo_data=geo_data,
        data=map_df,
        columns=['iso_a2', 'mean'],
        key_on='feature.properties.ISO3166-1-Alpha-2',
        fill_color='YlGn',
        fill_opacity=0.75,
        line_opacity=0.3,
        nan_fill_color='#2a2a2a',   # 데이터 없는 국가는 어두운 회색
        nan_fill_opacity=0.4,
        legend_name='평균 평점 (vote_average)',
        highlight=True              # 마우스 호버 시 해당 국가 강조
    ).add_to(m)

    # 호버 툴팁: 국가 위에 마우스 올리면 ISO 코드 + 국가명 표시
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['ISO3166-1-Alpha-2', 'name'],
            aliases=['국가코드', '국가명']
        )
    )

    # 데이터가 있는 국가에만 평점 + n수 레이블 마커 추가
    for feature in geo_data['features']:
        iso = feature['properties'].get('ISO3166-1-Alpha-2', '')
        row = map_df[map_df['iso_a2'] == iso]
        if row.empty:
            continue    # 분석 대상이 아닌 국가는 마커 생략

        # 국가 폴리곤의 중심점 계산 (위도, 경도 평균)
        # MultiPolygon이면 가장 큰 폴리곤 기준으로 계산
        try:
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
            elif geom['type'] == 'MultiPolygon':
                coords = max(geom['coordinates'], key=lambda p: len(p[0]))[0]  # 이 부분 LLM 사용
            else:
                continue
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            center = [sum(lats)/len(lats), sum(lons)/len(lons)]
        except Exception:
            continue

        mean_val = row['mean'].values[0]
        n_val = row['n'].values[0]
        lower = row['lower'].values[0]
        upper = row['upper'].values[0]
        name = row['country_name'].values[0]

        # DivIcon으로 HTML 레이블 마커 생성
        # 국가 중심점에 ISO 코드 + 평점 + n수 표시
        # 배경 검정 60% 불투명, 글자색 흰색, 글자크기 10, 글자 Bold, 박스 내부 여백 2,5px
        # 모서리 둥글게 / 반투명 흰색 테두리, 텍스트 줄바꿈 X, 가운데 정렬
        folium.Marker(
            location=center,
            icon=folium.DivIcon(
                html=f"""<div style="
                    background:rgba(0,0,0,0.6);
                    color:white;
                    font-size:10px;
                    font-weight:bold;
                    padding:2px 5px;
                    border-radius:4px;
                    border:1px solid rgba(255,255,255,0.2);
                    white-space:nowrap;
                    text-align:center;">
                    <span style="color:#90ee90">{iso}</span> {mean_val:.2f}<br>
                    <span style="font-size:9px;color:#aaa">n={n_val:,}</span>
                </div>""",
                # span: ISO 코드만 초록색(#90ee90)으로 강조, 뒤에 평점 표시
                # br: 줄바꿈
                # 두 번째 span: n수는 9px로 작게, 회색(#aaa)으로 표시
                icon_size=(75, 32),   # 마커 박스 크기 (가로 75px, 세로 32px)
                icon_anchor=(37, 16)  # 기준점: 박스 정중앙 (75/2=37, 32/2=16)
                                      # 이 기준점이 지도상 center 좌표에 고정됨
            ),
            # 클릭 시 팝업: 국가명 + 평균 평점 + 95% CI + 데이터 수
            popup=folium.Popup(
                f"<b>{name}</b><br>평균 평점: {mean_val:.4f}<br>"
                f"95% CI: [{lower:.4f}, {upper:.4f}]<br>데이터 수: {n_val:,}편",
                max_width=200
            )
        ).add_to(m)

    # 레이어 컨트롤 추가 (choropleth 레이어 on/off 가능)
    folium.LayerControl().add_to(m)

    # HTML 파일로 저장 후 기본 브라우저에서 바로 열기
    m.save('tmdbv11_hypothesis3_map.html')
    print("저장: tmdbv11_hypothesis3_map.html")
    webbrowser.open('file://' + os.path.abspath('tmdbv11_hypothesis3_map.html'))
    print("브라우저에서 지도 열림")


# main
if __name__ == "__main__":

    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess()

    grand_mean = df['vote_average'].mean()
    print(f"\n전체 평균 평점: {grand_mean:.4f}")

    # 2. 분석
    ci_df = analyze_hypothesis3(df)

    # 3. 에러바 그래프
    print("\n[에러바 그래프 생성 중...]")
    plot_errorbar(ci_df, grand_mean)

    # 4. folium 세계 지도
    print("\n[folium 세계 지도 생성 중...]")
    plot_folium_map(ci_df)
