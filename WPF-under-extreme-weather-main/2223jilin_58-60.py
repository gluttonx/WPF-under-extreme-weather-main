# ========================= 1. jilin_058 =========================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
import os

# 文件路径
excel_path = '2223jilin_058_processed_4classes.xlsx'
output_mat = '58wf_4_train.mat'

# 选的5个关键NWP特征（匹配原代码 nwp_index）
selected_features = [
    'wind_speed_100m', 'wind_direction_100m', 
    'temperature_2m', 'pressure_msl', 'relative_humidity_2m'
]
num_features = len(selected_features)
target_col = 'Power2'

# 常规天气聚类数
num_conven_classes = 10

# 极端天气sheet名（直接对应class1~4）
extreme_sheet_names = [
    'extreme_high_wind',   # class1
    'extreme_high_temp',   # class2
    'extreme_cold_wave',   # class3
    'extreme_frost'        # class4
]

# 读取数据
xls = pd.ExcelFile(excel_path)

# 全体数据（用于 p_1h / nwp_1h）
full_df = pd.read_excel(xls, sheet_name='jilin_058')

# 常规天气（用于聚类）
normal_df = pd.read_excel(xls, sheet_name='normal_weather')

# 极端天气（直接用，不聚类）
extreme_dfs = [pd.read_excel(xls, sheet_name=name) for name in extreme_sheet_names]

# ------------------ 修改位置：读取数据后，聚类前（插入以下代码块） ------------------
# 装机容量
Cap = 50.0

# 对所有相关DataFrame的Power2进行归一化（除以容量）
full_df[target_col] = full_df[target_col] / Cap
normal_df[target_col] = normal_df[target_col] / Cap
for ext_df in extreme_dfs:
    ext_df[target_col] = ext_df[target_col] / Cap

print(f"Power2 已归一化（除以容量 {Cap}）")
# ------------------ 修改结束（后续聚类和mat_dict生成保持不变） ------------------

# ------------------ 只对常规天气聚类 ------------------
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_df[selected_features])

kmeans = KMeans(n_clusters=num_conven_classes, random_state=42, n_init='auto')
labels = kmeans.fit_predict(normal_scaled)

normal_df['cluster_label'] = labels
conven_class_dfs = [normal_df[normal_df['cluster_label'] == i].drop(columns='cluster_label') 
                    for i in range(num_conven_classes)]

# ------------------ 生成 .mat dict ------------------
mat_dict = {}

# 全体数据
mat_dict['p_1h'] = full_df[[target_col]].values.reshape(-1, 1)  # (T, 1)
mat_dict['nwp_1h'] = full_df[selected_features].values  # (T, 5)

# 常规全体（代码有时用）
mat_dict['p_conven'] = normal_df[[target_col]].values.reshape(-1, 1)
mat_dict['nwp_conven_'] = normal_df[selected_features].values

# 常规聚类功率：p_conven_class (1x10 object)
mat_dict['p_conven_class'] = np.empty((1, num_conven_classes), dtype=object)
for i in range(num_conven_classes):
    mat_dict['p_conven_class'][0, i] = conven_class_dfs[i][[target_col]].values.reshape(-1, 1)

# 常规聚类NWP：nwp_conven_class_ (1x5 object, 内层1x10 object)
mat_dict['nwp_conven_class_'] = np.empty((1, num_features), dtype=object)
for f_idx, feat in enumerate(selected_features):
    feature_per_class = np.empty((1, num_conven_classes), dtype=object)
    for c in range(num_conven_classes):
        feature_per_class[0, c] = conven_class_dfs[c][[feat]].values.reshape(-1, 1)
    mat_dict['nwp_conven_class_'][0, f_idx] = feature_per_class

# 极端天气（直接用4个sheet，不聚类）
for idx in range(4):
    ext_df = extreme_dfs[idx]
    class_key_p = f'p_extre_class{idx+1}'
    class_key_nwp = f'nwp_extre_class{idx+1}_'
    
    # 功率
    mat_dict[class_key_p] = ext_df[[target_col]].values.reshape(-1, 1)
    
    # NWP：1x5 object，每个是该类该特征序列
    mat_dict[class_key_nwp] = np.empty((1, num_features), dtype=object)
    for f_idx, feat in enumerate(selected_features):
        mat_dict[class_key_nwp][0, f_idx] = ext_df[[feat]].values.reshape(-1, 1)

# 保存
savemat(output_mat, mat_dict)
print(f'生成完成：{output_mat}')
print('极端天气直接用了4个sheet（无聚类），常规聚类成10类，完全匹配原代码结构！')


# ========================= 2. jilin_059 =========================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
import os

# 文件路径
excel_path = '2223jilin_059_processed_4classes.xlsx'
output_mat = '59wf_4_train.mat'

# 选的5个关键NWP特征（匹配原代码 nwp_index）
selected_features = [
    'wind_speed_100m', 'wind_direction_100m', 
    'temperature_2m', 'pressure_msl', 'relative_humidity_2m'
]
num_features = len(selected_features)
target_col = 'Power2'

# 常规天气聚类数
num_conven_classes = 10

# 极端天气sheet名（直接对应class1~4）
extreme_sheet_names = [
    'extreme_high_wind',   # class1
    'extreme_high_temp',   # class2
    'extreme_cold_wave',   # class3
    'extreme_frost'        # class4
]

# 读取数据
xls = pd.ExcelFile(excel_path)

# 全体数据（用于 p_1h / nwp_1h）
full_df = pd.read_excel(xls, sheet_name='jilin_059')

# 常规天气（用于聚类）
normal_df = pd.read_excel(xls, sheet_name='normal_weather')

# 极端天气（直接用，不聚类）
extreme_dfs = [pd.read_excel(xls, sheet_name=name) for name in extreme_sheet_names]

# ------------------ 修改位置：读取数据后，聚类前（插入以下代码块） ------------------
# 装机容量
Cap = 50.0

# 对所有相关DataFrame的Power2进行归一化（除以容量）
full_df[target_col] = full_df[target_col] / Cap
normal_df[target_col] = normal_df[target_col] / Cap
for ext_df in extreme_dfs:
    ext_df[target_col] = ext_df[target_col] / Cap

print(f"Power2 已归一化（除以容量 {Cap}）")
# ------------------ 修改结束（后续聚类和mat_dict生成保持不变） ------------------

# ------------------ 只对常规天气聚类 ------------------
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_df[selected_features])

kmeans = KMeans(n_clusters=num_conven_classes, random_state=42, n_init='auto')
labels = kmeans.fit_predict(normal_scaled)

normal_df['cluster_label'] = labels
conven_class_dfs = [normal_df[normal_df['cluster_label'] == i].drop(columns='cluster_label') 
                    for i in range(num_conven_classes)]

# ------------------ 生成 .mat dict ------------------
mat_dict = {}

# 全体数据
mat_dict['p_1h'] = full_df[[target_col]].values.reshape(-1, 1)  # (T, 1)
mat_dict['nwp_1h'] = full_df[selected_features].values  # (T, 5)

# 常规全体（代码有时用）
mat_dict['p_conven'] = normal_df[[target_col]].values.reshape(-1, 1)
mat_dict['nwp_conven_'] = normal_df[selected_features].values

# 常规聚类功率：p_conven_class (1x10 object)
mat_dict['p_conven_class'] = np.empty((1, num_conven_classes), dtype=object)
for i in range(num_conven_classes):
    mat_dict['p_conven_class'][0, i] = conven_class_dfs[i][[target_col]].values.reshape(-1, 1)

# 常规聚类NWP：nwp_conven_class_ (1x5 object, 内层1x10 object)
mat_dict['nwp_conven_class_'] = np.empty((1, num_features), dtype=object)
for f_idx, feat in enumerate(selected_features):
    feature_per_class = np.empty((1, num_conven_classes), dtype=object)
    for c in range(num_conven_classes):
        feature_per_class[0, c] = conven_class_dfs[c][[feat]].values.reshape(-1, 1)
    mat_dict['nwp_conven_class_'][0, f_idx] = feature_per_class

# 极端天气（直接用4个sheet，不聚类）
for idx in range(4):
    ext_df = extreme_dfs[idx]
    class_key_p = f'p_extre_class{idx+1}'
    class_key_nwp = f'nwp_extre_class{idx+1}_'
    
    # 功率
    mat_dict[class_key_p] = ext_df[[target_col]].values.reshape(-1, 1)
    
    # NWP：1x5 object，每个是该类该特征序列
    mat_dict[class_key_nwp] = np.empty((1, num_features), dtype=object)
    for f_idx, feat in enumerate(selected_features):
        mat_dict[class_key_nwp][0, f_idx] = ext_df[[feat]].values.reshape(-1, 1)

# 保存
savemat(output_mat, mat_dict)
print(f'生成完成：{output_mat}')
print('极端天气直接用了4个sheet（无聚类），常规聚类成10类，完全匹配原代码结构！')


# ========================= 3. jilin_060 =========================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
import os

# 文件路径
excel_path = '2223jilin_060_processed_4classes.xlsx'
output_mat = '60wf_4_train.mat'

# 选的5个关键NWP特征（匹配原代码 nwp_index）
selected_features = [
    'wind_speed_100m', 'wind_direction_100m', 
    'temperature_2m', 'pressure_msl', 'relative_humidity_2m'
]
num_features = len(selected_features)
target_col = 'Power2'

# 常规天气聚类数
num_conven_classes = 10

# 极端天气sheet名（直接对应class1~4）
extreme_sheet_names = [
    'extreme_high_wind',   # class1
    'extreme_high_temp',   # class2
    'extreme_cold_wave',   # class3
    'extreme_frost'        # class4
]

# 读取数据
xls = pd.ExcelFile(excel_path)

# 全体数据（用于 p_1h / nwp_1h）
full_df = pd.read_excel(xls, sheet_name='jilin_060')

# 常规天气（用于聚类）
normal_df = pd.read_excel(xls, sheet_name='normal_weather')

# 极端天气（直接用，不聚类）
extreme_dfs = [pd.read_excel(xls, sheet_name=name) for name in extreme_sheet_names]

# ------------------ 修改位置：读取数据后，聚类前（插入以下代码块） ------------------
# 装机容量
Cap = 100.0

# 对所有相关DataFrame的Power2进行归一化（除以容量）
full_df[target_col] = full_df[target_col] / Cap
normal_df[target_col] = normal_df[target_col] / Cap
for ext_df in extreme_dfs:
    ext_df[target_col] = ext_df[target_col] / Cap

print(f"Power2 已归一化（除以容量 {Cap}）")
# ------------------ 修改结束（后续聚类和mat_dict生成保持不变） ------------------

# ------------------ 只对常规天气聚类 ------------------
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_df[selected_features])

kmeans = KMeans(n_clusters=num_conven_classes, random_state=42, n_init='auto')
labels = kmeans.fit_predict(normal_scaled)

normal_df['cluster_label'] = labels
conven_class_dfs = [normal_df[normal_df['cluster_label'] == i].drop(columns='cluster_label') 
                    for i in range(num_conven_classes)]

# ------------------ 生成 .mat dict ------------------
mat_dict = {}

# 全体数据
mat_dict['p_1h'] = full_df[[target_col]].values.reshape(-1, 1)  # (T, 1)
mat_dict['nwp_1h'] = full_df[selected_features].values  # (T, 5)

# 常规全体（代码有时用）
mat_dict['p_conven'] = normal_df[[target_col]].values.reshape(-1, 1)
mat_dict['nwp_conven_'] = normal_df[selected_features].values

# 常规聚类功率：p_conven_class (1x10 object)
mat_dict['p_conven_class'] = np.empty((1, num_conven_classes), dtype=object)
for i in range(num_conven_classes):
    mat_dict['p_conven_class'][0, i] = conven_class_dfs[i][[target_col]].values.reshape(-1, 1)

# 常规聚类NWP：nwp_conven_class_ (1x5 object, 内层1x10 object)
mat_dict['nwp_conven_class_'] = np.empty((1, num_features), dtype=object)
for f_idx, feat in enumerate(selected_features):
    feature_per_class = np.empty((1, num_conven_classes), dtype=object)
    for c in range(num_conven_classes):
        feature_per_class[0, c] = conven_class_dfs[c][[feat]].values.reshape(-1, 1)
    mat_dict['nwp_conven_class_'][0, f_idx] = feature_per_class

# 极端天气（直接用4个sheet，不聚类）
for idx in range(4):
    ext_df = extreme_dfs[idx]
    class_key_p = f'p_extre_class{idx+1}'
    class_key_nwp = f'nwp_extre_class{idx+1}_'
    
    # 功率
    mat_dict[class_key_p] = ext_df[[target_col]].values.reshape(-1, 1)
    
    # NWP：1x5 object，每个是该类该特征序列
    mat_dict[class_key_nwp] = np.empty((1, num_features), dtype=object)
    for f_idx, feat in enumerate(selected_features):
        mat_dict[class_key_nwp][0, f_idx] = ext_df[[feat]].values.reshape(-1, 1)

# 保存
savemat(output_mat, mat_dict)
print(f'生成完成：{output_mat}')
print('极端天气直接用了4个sheet（无聚类），常规聚类成10类，完全匹配原代码结构！')