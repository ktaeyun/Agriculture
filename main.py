# main.py
import numpy as np
import pandas as pd
import os
import glob
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from embedding_utils import time_delay_embedding
from tda_utils import compute_persistence_diagram_from_embedding, pd_distance
from plot_utils import plot_clusters
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

print("[INFO] Loading data...")

data_dir = "./dataset"
file_list = glob.glob(os.path.join(data_dir, "*.xlsx"))

filtered_results = []

# -------------------- A. 파일별 처리 --------------------
for file_path in sorted(file_list):
    df = pd.read_excel(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['평균가격'] = pd.to_numeric(df['평균가격'], errors='coerce').fillna(np.nan)
    varieties = df['품종'].unique()

    for variety in varieties:
        sub_df = df[df['품종'] == variety].copy().sort_values(by='DATE')
        valid_dates = sub_df.loc[~sub_df['평균가격'].isna(), 'DATE']

        if len(valid_dates) > 0:
            valid_first_date = valid_dates.min()
            valid_last_date = valid_dates.max()
        else:
            continue

        missing_count = sub_df['평균가격'].isna().sum()
        total_count = len(sub_df)
        missing_ratio = (missing_count / total_count) * 100 if total_count > 0 else 0

        if missing_ratio <= 70:
            filtered_results.append({
                'file_path': file_path,
                'variety': variety,
                'start_date': valid_first_date,
                'end_date': valid_last_date
            })

if not filtered_results:
    raise ValueError("조건 충족 품종 없음.")

# -------------------- B. 공통 구간 계산 --------------------
all_starts = [item['start_date'] for item in filtered_results]
all_ends = [item['end_date'] for item in filtered_results]
common_start = max(all_starts)
common_end = min(all_ends)

print("공통 구간:", common_start.date(), "~", common_end.date())

# -------------------- C. 유효 품종 데이터 준비 --------------------
X, y = [], []

for item in filtered_results:
    file_path = item['file_path']
    variety = item['variety']

    df = pd.read_excel(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['평균가격'] = pd.to_numeric(df['평균가격'], errors='coerce').fillna(0)

    sub_df = df[df['품종'] == variety]
    sub_df = sub_df[(sub_df['DATE'] >= common_start) & (sub_df['DATE'] <= common_end)]
    series = sub_df['평균가격'].values

    if len(series) < 2:
        continue

    X.append(series.reshape(1, -1))
    y.append(variety)

if not X:
    raise ValueError("공통 구간 자르고 나서 유효한 시리즈 없음.")

# -------------------- D. 길이 맞추기 --------------------
max_len = max([series.shape[1] for series in X])
X_padded = []
for series in X:
    current_len = series.shape[1]
    if current_len < max_len:
        padded = np.hstack([series, np.zeros((1, max_len - current_len))])
    else:
        padded = series
    X_padded.append(padded)

X = np.vstack(X_padded)
y = np.array(y)

print("시리즈 개수:", len(X))
print("품종:", np.unique(y))

# -------------------- E. 정규화 --------------------
X = TimeSeriesScalerMeanVariance().fit_transform(X)

# -------------------- F. Embedding & persistence diagram --------------------
print("[INFO] Computing embeddings and diagrams...")

diagrams = []
for i in range(len(X)):
    embedded = time_delay_embedding(X[i].flatten(), dimension=3, delay=1)
    diag = compute_persistence_diagram_from_embedding(embedded)
    diagrams.append(diag)

# -------------------- G. Distance matrix --------------------
print("[INFO] Computing pairwise distance matrix...")
num_series = len(diagrams)
dist_matrix = np.zeros((num_series, num_series))

for i in range(num_series):
    for j in range(i + 1, num_series):
        dist = pd_distance(diagrams[i], diagrams[j])
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

# -------------------- H. t-SNE embedding --------------------
tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
X_embedded = tsne.fit_transform(dist_matrix)

# -------------------- I. Clustering --------------------
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_embedded)

# -------------------- J. Visualization --------------------
plot_clusters(X_embedded, labels, title="TDA-based clustering (Persistence diagram distance)")

# -------------------- K. Save results --------------------
result_df = pd.DataFrame({'Variety': y, 'Cluster': labels})
print(result_df.sort_values(by='Cluster'))
result_df.to_csv("clustering_result_tda.csv", index=False, encoding='utf-8-sig')
print("[INFO] Clustering result saved to clustering_result_tda.csv")
print("[INFO] Done.")
