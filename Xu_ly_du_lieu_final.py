import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Đọc dữ liệu từ file CSV
file_path = 'Intel_CPUs.csv'
data = pd.read_csv(file_path)

# Chọn các cột cần thiết
columns_to_extract = [
    'Product_Collection', 'Recommended_Customer_Price',
    'nb_of_Cores', 'nb_of_Threads', 'Cache', 'Bus_Speed',
    'Max_Memory_Size', 'Max_Memory_Bandwidth'
]
data = data[columns_to_extract]

# Hàm chuẩn hóa cột Product_Collection
def clean_product_collection(value):
    if 'X-series' in value:
        return 'X-series'
    elif 'Celeron' in value:
        return 'Celeron'
    elif 'Pentium' in value:
        return 'Pentium'
    elif 'Quark' in value:
        return 'Quark'
    elif 'Atom' in value:
        return 'Atom'
    elif 'Itanium' in value:
        return 'Itanium'
    elif 'Xeon' in value:
        return 'Xeon'
    elif 'Core m' in value:
        return 'm'
    elif 'Core' in value:
        return 'Core'
    else:
        return value

data['Product_Collection'] = data['Product_Collection'].apply(clean_product_collection)

# Hàm làm sạch cột Recommended_Customer_Price
def clean_price(value):
    if isinstance(value, str):
        value = value.replace(',', '')
        if '-' in value:
            prices = [float(p.replace('$', '').strip()) for p in value.split('-')]
            return sum(prices) / len(prices)
        return float(value.replace('$', '').strip())
    return value

data['Recommended_Customer_Price'] = data['Recommended_Customer_Price'].apply(clean_price)

# Thay thế giá trị NaN bằng giá trị trung bình của Product_Collection
mean_prices_by_product = data.groupby('Product_Collection')['Recommended_Customer_Price'].mean()

def fill_missing_price(row):
    if pd.isna(row['Recommended_Customer_Price']):
        return mean_prices_by_product[row['Product_Collection']]
    return row['Recommended_Customer_Price']

data['Recommended_Customer_Price'] = data.apply(fill_missing_price, axis=1)

# Hàm làm sạch và chuyển đổi giá trị cột số học
def clean_numeric_column(value, unit_to_remove, conversion_factor=1):
    if isinstance(value, str):
        value = value.replace(unit_to_remove, '').strip()
        if value == '' or value.lower() == 'n/a':
            return np.nan
    return pd.to_numeric(value, errors='coerce') * conversion_factor

# Xử lý cột Max_Memory_Bandwidth
data['Max_Memory_Bandwidth'] = data['Max_Memory_Bandwidth'].apply(lambda x: clean_numeric_column(x, ' GB/s'))
data['Max_Memory_Bandwidth'] = data['Max_Memory_Bandwidth'].fillna(data['Max_Memory_Bandwidth'].mean())

# Hàm làm sạch và chuyển đổi giá trị trong cột Bus_Speed
def clean_bus_speed(value):
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() == 'n/a':
            return np.nan
        if 'MHz' in value:
            try:
                return float(value.replace('MHz', '').strip()) * 1000
            except ValueError:
                return np.nan
        if 'GT/s' in value:
            try:
                clean_value = value.replace('GT/s', '').replace('DMI', '').replace('QPI', '').replace('OPI', '').strip()
                base_value = float(clean_value)
                return base_value * 500000
            except ValueError:
                return np.nan
    return pd.to_numeric(value, errors='coerce')

data['Bus_Speed'] = data['Bus_Speed'].apply(clean_bus_speed)
mean_bus_speed_by_product = data.groupby('Product_Collection')['Bus_Speed'].mean()

def fill_missing_bus_speed(row):
    if pd.isna(row['Bus_Speed']):
        return mean_bus_speed_by_product[row['Product_Collection']]
    return row['Bus_Speed']

data['Bus_Speed'] = data.apply(fill_missing_bus_speed, axis=1)
data['Bus_Speed'] = data['Bus_Speed'].fillna(data['Bus_Speed'].mean())

# Hàm làm sạch và chuyển đổi giá trị trong cột Cache
def clean_cache(value):
    if isinstance(value, str):
        value = value.lower().replace('smartcache', '').replace('l2', '').replace('l3', '').strip()
        if 'mb' in value:
            try:
                return float(value.replace('mb', '').strip()) * 1024
            except ValueError:
                return np.nan
        elif 'kb' in value:
            try:
                return float(value.replace('kb', '').strip())
            except ValueError:
                return np.nan
        elif value == '' or value == 'n/a':
            return np.nan
    return pd.to_numeric(value, errors='coerce')

data['Cache'] = data['Cache'].apply(clean_cache)
data['Cache'] = data['Cache'].fillna(data['Cache'].mean())

# Xử lý cột nb_of_Threads
data['nb_of_Threads'] = data['nb_of_Threads'].fillna(data['nb_of_Threads'].mean())

# Hàm làm sạch và chuyển đổi giá trị trong cột Max_Memory_Size
def clean_max_memory_size(value):
    if isinstance(value, str):
        value = value.strip()
        if 'TB' in value:
            try:
                return float(value.replace('TB', '').strip()) * 1024
            except ValueError:
                return np.nan
        elif 'GB' in value:
            try:
                return float(value.replace('GB', '').strip())
            except ValueError:
                return np.nan
        elif value == '' or value.lower() == 'n/a':
            return np.nan
    return pd.to_numeric(value, errors='coerce')

data['Max_Memory_Size'] = data['Max_Memory_Size'].apply(clean_max_memory_size)
data['Max_Memory_Size'] = data['Max_Memory_Size'].fillna(data['Max_Memory_Size'].mean())

# Biến đổi dữ liệu nếu cần thiết
data['Log_Recommended_Customer_Price'] = np.log1p(data['Recommended_Customer_Price'])
data['Log_Max_Memory_Bandwidth'] = np.log1p(data['Max_Memory_Bandwidth'])

# Thêm biến bậc cao hoặc tương tác
data['nb_of_Cores_squared'] = data['nb_of_Cores'] ** 2
data['Bus_Speed_squared'] = data['Bus_Speed'] ** 2
data['Interaction_Term'] = data['nb_of_Cores'] * data['Bus_Speed']

# Danh sách biến độc lập
X = data[['nb_of_Cores', 'nb_of_Threads', 'Cache', 'Bus_Speed', 'Max_Memory_Size', 
          'Log_Max_Memory_Bandwidth', 'nb_of_Cores_squared', 'Bus_Speed_squared', 'Interaction_Term']]
y = data['Log_Recommended_Customer_Price']

# Thêm cột hệ số chặn cho mô hình hồi quy
X = sm.add_constant(X)

# Tính toán VIF
def calculate_vif(X):
    """
    Hàm tính toán Variance Inflation Factor (VIF) cho các biến độc lập.
    
    Parameters:
    X (pd.DataFrame): DataFrame chứa các biến độc lập (không bao gồm biến phụ thuộc).
    
    Returns:
    pd.DataFrame: DataFrame chứa tên biến và giá trị VIF tương ứng.
    """
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Tính VIF và in kết quả
vif_result = calculate_vif(X)
print("\nVariance Inflation Factor (VIF):")
print(vif_result)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính với statsmodels
model = sm.OLS(y_train, X_train).fit()

# In kết quả tóm tắt của mô hình
print(model.summary())

# Dự đoán giá trị trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán phần dư
residuals = y_test - y_pred

# Biểu đồ phần dư so với giá trị dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs Fitted Values (Adjusted)', fontsize=16)
plt.xlabel('Fitted Values (Predicted)', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# Histogram của phần dư
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Residuals', fontsize=16)
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# Q-Q Plot để kiểm tra phân phối chuẩn
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Adjusted)', fontsize=16)
plt.grid(alpha=0.3)
plt.show()