# 영화 데이터를 활용하여 영화 흥행에 영향을 미치는 주요 원인 분석


### 가설
 가설 1 제작비가 높을수록 수익은 증가할 것이다 | 지지 (r=0.6383)  
 가설 2 평점이 높을수록 수익은 증가할 것이다 | 기각 (r=0.1524)  
 가설 3 제작 국가에 따라 평점에 유의미한 차이가 있을 것이다 | 지지 (상하위 5개국 신뢰구간 겹치지 않음)  

---

## 파일 구조

```
.
├── preprocessing.py          # 전처리 공통 모듈 (가설 1, 2에서 import)
├── hypothesis1.py            # 가설 1 분석 및 시각화
├── hypothesis2.py            # 가설 2 분석 및 시각화
├── tmdbv11_hypothesis3.py    # 가설 3 분석 및 시각화 (직접 다운로드 필요)
├── tmdb_5000_movies.csv      # TMDB 5000 영화 데이터 (가설 1, 2)
└── movie_budgets_and_revenues.csv  # The Numbers 재무 데이터 (가설 1, 2)
```

---

## 데이터

### 가설 1, 2
- **TMDB 5000 Movies Dataset**: [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **The Numbers - Movie Budgets & Revenue**: [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/the-numbers-movie-budgets-and-revenues)

### 가설 3
- **TMDB Movies Dataset v11**: [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
- 파일 크기(약 300MB)로 인해 GitHub push 불가 > 위 링크에서 직접 다운로드 후 `tmdbv11_hypothesis3.py`와 같은 폴더에 위치시킬 것
- 파일명: `TMDB_movie_dataset_v11.csv`

---

## 설치 및 실행

```bash
pip install pandas numpy matplotlib scipy folium
```

### 가설 1 실행
```bash
python hypothesis1.py
```

### 가설 2 실행
```bash
python hypothesis2.py
```

### 가설 3 실행
```bash
python tmdbv11_hypothesis3.py
```

---

## 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| pandas | 데이터 로드 및 전처리 |
| numpy | 수치 계산 |
| matplotlib | 에러바 그래프, 산점도 시각화 |
| scipy | 피어슨 상관분석, 선형회귀 |
| folium | 세계 지도 choropleth 시각화 (가설 3) |

최신 버전 설치 시 정상 동작합니다.
구 라이브러리에서 parsing 에러 확인 (pandas)

---

## GitHub

[https://github.com/B117-1-Team-8/Analyze-TMDB-Movie](https://github.com/B117-1-Team-8/Analyze-TMDB-Movie)