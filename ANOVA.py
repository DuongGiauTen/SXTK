from scipy.stats import shapiro
from scipy.stats import levene
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Đặt mã hóa UTF-8 cho đầu ra
import pandas as pd  # type: ignore
import numpy as np
# Đường dẫn tới file Intel_CPUs.csv
file_path = r"C:\Study\xstk\Intel_CPUs.csv"

# Đọc dữ liệu từ file CSV
data = pd.read_csv(file_path)

# Tùy chỉnh hiển thị
pd.set_option('display.max_rows', 25)  # Hiển thị tối đa 25 dòng
pd.set_option('display.max_columns', 5)  # Hiển thị 5 cột
pd.set_option('display.width', 500)  # Tăng chiều rộng hiển thị để tránh cột bị cắt
# Hiển thị dữ liệu
print(data.head(25))  # Hiển thị 25 dòng đầu tiên
# Hiển thị thông tin chi tiết về dữ liệu
print(data.info())

# Trích xuất các cột quan trọng
columns_to_extract = [
    'Product_Collection', 'Launch_Date', 'Recommended_Customer_Price',
    'nb_of_Cores', 'nb_of_Threads', 'Cache', 'Bus_Speed',
    'Processor_Base_Frequency', 'Max_Memory_Size', 'Max_Memory_Bandwidth'
]
du_lieu_moi = data[columns_to_extract]

# Hiển thị 20 dòng đầu tiên của dữ liệu đã trích xuất
print(du_lieu_moi.head(20))

# Thống kê các giá trị bị khuyết của du_lieu_moi
missing_values = du_lieu_moi.isnull().sum()
print("\nSố lượng giá trị bị khuyết trong mỗi cột:")
print(missing_values)
print(du_lieu_moi[['Product_Collection','Launch_Date', 'Recommended_Customer_Price']].head(20))
du_lieu_moi = du_lieu_moi.dropna(subset=['Launch_Date'])  # Loại bỏ các dòng có Launch_Date bị khuyết

# Định dạng lại cột Launch_Date theo công thức: Số năm + 1 + Số quý
def convert_launch_date(date):
    quarter_mapping = {'1': 0.00, '2': 0.25, '3': 0.50, '4': 0.75}
    quarter = quarter_mapping[date[1]]  # Lấy giá trị quý
    year = int(date[3:].replace("'", "")) + 0  # Lấy năm và chuyển thành số nguyên đầy đủ
    result = year + 1 + quarter
    # Nếu kết quả vượt quá 100, trừ đi 100
    if result >= 100:
        result = result - 100
    return result

du_lieu_moi['Launch_Date'] = du_lieu_moi['Launch_Date'].apply(convert_launch_date)

# Chuẩn hóa cột Product_Collection
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

du_lieu_moi['Product_Collection'] = du_lieu_moi['Product_Collection'].apply(clean_product_collection)

# Hiển thị dữ liệu đã xử lý
print(du_lieu_moi.head(20))
# Xử lý cột Recommended_Customer_Price
def clean_price(value):
    if isinstance(value, str):
        value = value.replace(',', '')  # Loại bỏ dấu phẩy
        if '-' in value:  # Nếu giá trị là khoảng (ví dụ: "$68-$72")
            prices = [float(p.replace('$', '').strip()) for p in value.split('-')]
            return sum(prices) / len(prices)  # Tính trung bình giá trị trong khoảng
        return float(value.replace('$', '').strip())  # Loại bỏ ký hiệu $ và chuyển thành float
    return value

# Áp dụng hàm xử lý cho cột Recommended_Customer_Price
du_lieu_moi['Recommended_Customer_Price'] = du_lieu_moi['Recommended_Customer_Price'].apply(clean_price)

# Tính giá trị trung bình của Recommended_Customer_Price theo từng loại Product_Collection
mean_prices_by_product = du_lieu_moi.groupby('Product_Collection')['Recommended_Customer_Price'].mean()

# Thay thế các giá trị NaN bằng giá trị trung bình tương ứng với Product_Collection
def fill_missing_price(row):
    if pd.isna(row['Recommended_Customer_Price']):  # Nếu giá trị bị khuyết
        return mean_prices_by_product[row['Product_Collection']]  # Thay bằng giá trị trung bình tương ứng
    return row['Recommended_Customer_Price']

du_lieu_moi['Recommended_Customer_Price'] = du_lieu_moi.apply(fill_missing_price, axis=1)

# Hiển thị dữ liệu đã xử lý
print(du_lieu_moi[['Product_Collection','Launch_Date', 'Recommended_Customer_Price']].head(20))


#Xử lý các cột Cache, Bus_Speed, Max_Memory_Size, Max_Memory_Bandwidth

# Hàm chuyển đổi giá trị thành số, loại bỏ đơn vị và xử lý giá trị không hợp lệ
def clean_numeric_column(value, unit_to_remove, conversion_factor=1):
    if isinstance(value, str):
        value = value.replace(unit_to_remove, '').strip()  # Loại bỏ đơn vị
        if value == '' or value.lower() == 'n/a':  # Xử lý chuỗi rỗng hoặc 'N/A'
            return np.nan
    return pd.to_numeric(value, errors='coerce') * conversion_factor  # Chuyển đổi và nhân với hệ số

# Xử lý cột Max_Memory_Bandwidth
du_lieu_moi['Max_Memory_Bandwidth'] = du_lieu_moi['Max_Memory_Bandwidth'].apply(
    lambda x: clean_numeric_column(x, ' GB/s')
)
du_lieu_moi['Max_Memory_Bandwidth'] = du_lieu_moi['Max_Memory_Bandwidth'].fillna(
    du_lieu_moi['Max_Memory_Bandwidth'].mean()
)  # Thay NaN bằng giá trị trung bình

print(du_lieu_moi[['Bus_Speed']].head(20))

# Hàm làm sạch và chuyển đổi giá trị trong cột Bus_Speed
def clean_bus_speed(value):
    if isinstance(value, str):
        value = value.strip()  # Loại bỏ khoảng trắng thừa
        
        # Xử lý các trường hợp đặc biệt
        if value == '' or value.lower() == 'n/a':  
            return np.nan
            
        # Xử lý đơn vị MHz
        if 'MHz' in value:
            try:
                return float(value.replace('MHz', '').strip()) * 1000  # 1 MHz = 1000 KHz
            except ValueError:  
                return np.nan
                
        # Xử lý các giao thức DMI/QPI/OPI với đơn vị GT/s
        if 'GT/s' in value:
            try:
                # Loại bỏ các ký tự không cần thiết và giữ lại số
                clean_value = value.replace('GT/s', '').replace('DMI', '').replace('QPI', '').replace('OPI', '').strip()
                base_value = float(clean_value)
                
                # Chuyển đổi dựa trên giao thức
                if 'DMI' in value:
                    return base_value * 500000  # DMI: 1 GT/s = 500000 KHz
                elif 'QPI' in value:
                    return base_value * 500000  # QPI: 1 GT/s = 500000 KHz
                elif 'OPI' in value:
                    return base_value * 500000  # OPI: 1 GT/s = 500000 KHz
                else:
                    return base_value * 500000  # Mặc định cho GT/s
            except ValueError:
                return np.nan
                
    return pd.to_numeric(value, errors='coerce')  # Chuyển đổi thành số hoặc NaN nếu không hợp lệ

# Áp dụng hàm làm sạch cho cột Bus_Speed trước
du_lieu_moi['Bus_Speed'] = du_lieu_moi['Bus_Speed'].apply(clean_bus_speed)

# Tính giá trị trung bình của Bus_Speed theo từng loại Product_Collection
mean_bus_speed_by_product = du_lieu_moi.groupby('Product_Collection')['Bus_Speed'].mean()

# Thay thế các giá trị NaN bằng giá trị trung bình tương ứng với Product_Collection
def fill_missing_bus_speed(row):
    if pd.isna(row['Bus_Speed']):  # Nếu giá trị bị khuyết
        return mean_bus_speed_by_product[row['Product_Collection']]  # Thay bằng giá trị trung bình của loại CPU tương ứng
    return row['Bus_Speed']

# Áp dụng hàm điền giá trị trống
du_lieu_moi['Bus_Speed'] = du_lieu_moi.apply(fill_missing_bus_speed, axis=1)

# Thay thế các giá trị NaN bằng giá trị trung bình của cột
du_lieu_moi['Bus_Speed'] = du_lieu_moi['Bus_Speed'].fillna(du_lieu_moi['Bus_Speed'].mean())

# Hiển thị dữ liệu đã xử lý
print("\nDữ liệu sau khi xử lý cột Bus_Speed:")
print(du_lieu_moi[['Bus_Speed']].head(20))
print(du_lieu_moi.isnull().sum())

# Hàm làm sạch và chuyển đổi giá trị trong cột Processor_Base_Frequency
def clean_processor_frequency(value):
    if isinstance(value, str):
        value = value.strip()  # Loại bỏ khoảng trắng thừa
        if 'GHz' in value:  # Nếu đơn vị là GHz, chuyển đổi sang MHz
            try:
                return float(value.replace('GHz', '').strip()) * 1000
            except ValueError:
                return np.nan
        elif 'MHz' in value:  # Nếu đơn vị là MHz, giữ nguyên giá trị
            try:
                return float(value.replace('MHz', '').strip())
            except ValueError:
                return np.nan
    return pd.to_numeric(value, errors='coerce')  # Chuyển đổi thành số hoặc NaN nếu không hợp lệ

# Áp dụng hàm làm sạch cho cột Processor_Base_Frequency
du_lieu_moi['Processor_Base_Frequency'] = du_lieu_moi['Processor_Base_Frequency'].apply(clean_processor_frequency)

# Thay thế các giá trị NaN bằng giá trị trung bình của cột
du_lieu_moi['Processor_Base_Frequency'] = du_lieu_moi['Processor_Base_Frequency'].fillna(
    du_lieu_moi['Processor_Base_Frequency'].mean()
)

# Hiển thị dữ liệu đã xử lý
print("\nDữ liệu sau khi xử lý cột Processor_Base_Frequency:")
print(du_lieu_moi[['Processor_Base_Frequency']].head(20))

# Kiểm tra xem còn giá trị bị khuyết hay không
missing_values_after = du_lieu_moi['Processor_Base_Frequency'].isnull().sum()
print(f"\nSố lượng giá trị bị khuyết trong cột Processor_Base_Frequency sau khi xử lý: {missing_values_after}")

# Hàm làm sạch và chuyển đổi giá trị trong cột Cache
def clean_cache(value):
    if isinstance(value, str):
        value = value.lower().replace('smartcache', '').replace('l2', '').replace('l3', '').strip()  # Loại bỏ ký tự dư thừa
        if 'mb' in value:  # Nếu đơn vị là MB, chuyển đổi sang KB
            try:
                return float(value.replace('mb', '').strip()) * 1024
            except ValueError:
                return np.nan
        elif 'kb' in value:  # Nếu đơn vị là KB, giữ nguyên giá trị
            try:
                return float(value.replace('kb', '').strip())
            except ValueError:
                return np.nan
        elif value == '' or value == 'n/a':  # Xử lý giá trị rỗng hoặc 'N/A'
            return np.nan
    return pd.to_numeric(value, errors='coerce')  # Chuyển đổi thành số hoặc NaN nếu không hợp lệ

# Áp dụng hàm làm sạch cho cột Cache
du_lieu_moi['Cache'] = du_lieu_moi['Cache'].apply(clean_cache)

# Thay thế các giá trị NaN bằng giá trị trung bình của cột
du_lieu_moi['Cache'] = du_lieu_moi['Cache'].fillna(du_lieu_moi['Cache'].mean())

# Hiển thị dữ liệu đã xử lý

# Kiểm tra xem còn giá trị bị khuyết hay không
missing_values_after = du_lieu_moi['Cache'].isnull().sum()
print(f"\nSố lượng giá trị bị khuyết trong cột Cache sau khi xử lý: {missing_values_after}")

print(du_lieu_moi[['Cache']].head(500))
print(du_lieu_moi.head(20))
print(du_lieu_moi.isnull().sum())

# Xử lý cột nb_of_Threads
du_lieu_moi['nb_of_Threads'] = du_lieu_moi['nb_of_Threads'].fillna(du_lieu_moi['nb_of_Threads'].mean())

# Hàm làm sạch và chuyển đổi giá trị trong cột Max_Memory_Size
def clean_max_memory_size(value):
    if isinstance(value, str):
        value = value.strip()  # Loại bỏ khoảng trắng thừa
        if 'TB' in value:  # Nếu đơn vị là TB, chuyển đổi sang GB
            try:
                return float(value.replace('TB', '').strip()) * 1024  # 1 TB = 1024 GB
            except ValueError:
                return np.nan
        elif 'GB' in value:  # Nếu đơn vị là GB, giữ nguyên giá trị
            try:
                return float(value.replace('GB', '').strip())
            except ValueError:
                return np.nan
        elif value == '' or value.lower() == 'n/a':  # Xử lý chuỗi rỗng hoặc 'N/A'
            return np.nan
    return pd.to_numeric(value, errors='coerce')  # Chuyển đổi thành số hoặc NaN nếu không hợp lệ

# Áp dụng hàm làm sạch cho cột Max_Memory_Size
du_lieu_moi['Max_Memory_Size'] = du_lieu_moi['Max_Memory_Size'].apply(clean_max_memory_size)

# Thay thế các giá trị NaN bằng giá trị trung bình của cột Max_Memory_Size
du_lieu_moi['Max_Memory_Size'] = du_lieu_moi['Max_Memory_Size'].fillna(du_lieu_moi['Max_Memory_Size'].mean())

# Hiển thị dữ liệu đã xử lý
print("\nDữ liệu sau khi xử lý cột nb_of_Threads và Max_Memory_Size:")
print(du_lieu_moi[['nb_of_Threads', 'Max_Memory_Size']].head(20))

# Kiểm tra xem còn giá trị bị khuyết hay không
missing_values_after = du_lieu_moi[['nb_of_Threads', 'Max_Memory_Size']].isnull().sum()
print(f"\nSố lượng giá trị bị khuyết trong các cột sau khi xử lý:\n{missing_values_after}")

# Kiểm tra kết quả
print(du_lieu_moi.head(50))
print(du_lieu_moi.isnull().sum())



# Tính các giá trị đặc trưng cho các biến định lượng
quantitative_columns = [
    'Launch_Date', 'Recommended_Customer_Price', 'nb_of_Cores', 'nb_of_Threads',
    'Processor_Base_Frequency', 'Cache', 'Bus_Speed', 'Max_Memory_Size', 'Max_Memory_Bandwidth'
]

# Tạo DataFrame chứa các giá trị đặc trưng
stats_df = pd.DataFrame({
    'mean': du_lieu_moi[quantitative_columns].mean(),
    'sd': du_lieu_moi[quantitative_columns].std(),
    'median': du_lieu_moi[quantitative_columns].median(),
    'Q1': du_lieu_moi[quantitative_columns].quantile(0.25),
    'Q3': du_lieu_moi[quantitative_columns].quantile(0.75),
    'min': du_lieu_moi[quantitative_columns].min(),
    'max': du_lieu_moi[quantitative_columns].max()
})

# Thống kê số lượng cho biến phân loại Product_Collection
product_collection_counts = du_lieu_moi['Product_Collection'].value_counts()
# Hiển thị số lượng cho biến phân loại Product_Collection
print("\nThống kê số lượng cho Product_Collection:")
print(product_collection_counts)



# Định dạng hiển thị số thập phân với 2 chữ số sau dấu phẩy
pd.options.display.float_format = '{:.2f}'.format
print(du_lieu_moi['Bus_Speed'].head(20))
# Hiển thị lại bảng thống kê các giá trị đặc trưng
print("\nBảng thống kê các giá trị đặc trưng :")
pd.set_option('display.max_columns', None)  # Hiển thị tất cả các cột
print(stats_df)
# Hiển thị số lượng cho biến phân loại Product_Collection
print("\nThống kê số lượng cho Product_Collection:")
print(product_collection_counts)












# Lọc các sản phẩm có số lượng >= 50
valid_products = product_collection_counts[product_collection_counts >= 50].index
# Giữ lại các dòng có Product_Collection nằm trong danh sách valid_products
du_lieu_moi = du_lieu_moi[du_lieu_moi['Product_Collection'].isin(valid_products)]
# Hiển thị lại thống kê sau khi lọc
print("\nThống kê số lượng cho Product_Collection sau khi lọc:")
print(du_lieu_moi['Product_Collection'].value_counts())










#vẽ biểu đồ QQ-plot
product_groups = du_lieu_moi['Product_Collection'].unique()
for group in product_groups:
    print(f"\nNhóm CPU: {group}")
    
    # Lọc dữ liệu Cache cho nhóm hiện tại
    group_data = du_lieu_moi[du_lieu_moi['Product_Collection'] == group]['Cache'].dropna()
    
    # Kiểm tra nếu dữ liệu rỗng hoặc không đủ
    if group_data.empty:
        print("Không có dữ liệu để kiểm định.")
        continue
    if len(group_data) < 3:
        print("Dữ liệu không đủ để thực hiện kiểm định.")
        continue
    
    # Kiểm định Shapiro-Wilk
    stat, p_value = shapiro(group_data)
    print(f"Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p_value:.2e}")
    
    if p_value > 0.05:
        print("Kết luận: Dữ liệu tuân theo phân phối chuẩn.")
    else:
        print("Kết luận: Dữ liệu không tuân theo phân phối chuẩn.")
 # Vẽ biểu đồ QQ-plot
    plt.figure(figsize=(6, 6))
    sm.qqplot(group_data, line='s')
    plt.title(f"QQ-Plot cho nhóm {group}")
    plt.show()










# Hàm kiểm định Levene và hiển thị boxplot
def perform_levene_test_with_boxplot(data):
    """
    Thực hiện kiểm định Levene để kiểm tra tính đồng nhất phương sai của cột Cache
    giữa các nhóm Product_Collection và hiển thị boxplot.

    Parameters:
        data (pd.DataFrame): Dữ liệu đầu vào.

    Returns:
        None: In ra kết quả kiểm định và hiển thị biểu đồ.
    """
    # Tách dữ liệu thành các nhóm dựa trên Product_Collection
    groups = [group['Cache'].dropna() for _, group in data.groupby('Product_Collection')]
    
    # Thực hiện kiểm định Levene
    stat, p_value = levene(*groups)
    
    # Tạo bảng kết quả
    results = pd.DataFrame({
        "Df": [len(groups) - 1, len(data) - len(groups)],
        "F Value": [stat],
        "Pr(>F)": [p_value]
    }, index=["group", "residual"])
    
    # In kết quả
    results["Pr(>F)"] = results["Pr(>F)"].apply(lambda x: f"{x:.2e}")  # Hiển thị 16 chữ số thập phân
    print(results)
    print("\n---")
    print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
    
    # Đánh giá kết quả kiểm định
    if p_value < 0.05:
        print("\nKết luận: Có sự khác biệt đáng kể về phương sai giữa các nhóm (p < 0.05).")
    else:
        print("\nKết luận: Không có sự khác biệt đáng kể về phương sai giữa các nhóm (p >= 0.05).")
    
    # Vẽ biểu đồ boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Product_Collection', y='Cache', data=data)
    plt.title("Boxplot of Cache by Product Collection")
    plt.xlabel("Product Collection")
    plt.ylabel("Cache (KB)")
    plt.xticks(rotation=45)
    plt.show()
# Gọi hàm kiểm định Levene và hiển thị boxplot
perform_levene_test_with_boxplot(du_lieu_moi)














# Chuẩn bị dữ liệu cho từng nhóm
groups = [group['Cache'].values for _, group in du_lieu_moi.groupby('Product_Collection')]

# Thực hiện kiểm định ANOVA
f_stat, p_value = f_oneway(*groups)

# Tính Mean Squares (Mean Sq) thủ công
sum_sq_between = sum([len(group) * (group.mean() - du_lieu_moi['Cache'].mean())**2 for group in groups])
sum_sq_within = sum([(value - group.mean())**2 for group in groups for value in group])
df_between = len(groups) - 1
df_within = len(du_lieu_moi) - len(groups)
mean_sq_between = sum_sq_between / df_between
mean_sq_within = sum_sq_within / df_within

# Tạo bảng kết quả ANOVA
anova_table = pd.DataFrame({
    'sum_sq': [sum_sq_between, sum_sq_within],
    'df': [df_between, df_within],
    'mean_sq': [mean_sq_between, mean_sq_within],
    'F': [f_stat, None],
    'PR(>F)': [p_value, None]
}, index=['Product_Collection', 'Residual'])

# Thêm cột Signif. codes dựa trên giá trị p-value
def signif_codes(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '.'
    else:
        return ' '

anova_table['Signif.'] = [signif_codes(p_value), None]

# Định dạng hiển thị: Số thực lớn hiển thị ở dạng khoa học, số nhỏ hiển thị bình thường
def custom_float_format(x):
    if isinstance(x, float) and (x > 1e5 or x < 1e-3):  # Chỉ áp dụng dạng khoa học cho số lớn hoặc rất nhỏ
        return f"{x:.2e}"
    elif isinstance(x, float):  # Hiển thị số thực nhỏ bình thường
        return f"{x:.2f}"
    else:  # Giữ nguyên số nguyên
        return str(x)

# Áp dụng định dạng tùy chỉnh cho DataFrame
pd.set_option('display.float_format', custom_float_format)

# Hiển thị kết quả
print("Kết quả phân tích phương sai (ANOVA):")
print(anova_table)

# Thêm phần giải thích Signif. codes
print("\n---")
print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")














#hàm kiểm tra Tuskey HSD
def perform_tukey_hsd(data, group_column, value_column):
    """
    Thực hiện kiểm tra Tukey HSD để kiểm tra sự khác biệt giữa các nhóm.
    Parameters:
        data (pd.DataFrame): Dữ liệu đầu vào.
        group_column (str): Tên cột chứa các nhóm (biến phân loại).
        value_column (str): Tên cột chứa giá trị định lượng cần kiểm tra.

    Returns:
        None: In ra kết quả kiểm tra Tukey HSD cho từng cặp nhóm.
    """
    # Kiểm tra Tukey HSD
    tukey = pairwise_tukeyhsd(endog=data[value_column], groups=data[group_column], alpha=0.05)
     # Hàm định dạng tùy chỉnh
    def format_value(value):
        if value < 0.001:
            return f"{value:.2e}"  # Hiển thị dạng số khoa học nếu nhỏ hơn 0.0001
        else:
            return f"{value:.4f}"  # Hiển thị dạng số thập phân nếu lớn hơn hoặc bằng 0.0001


    # In kết quả
    print("\nKết quả kiểm tra Tukey HSD:")
    print(tukey.summary())  # Hiển thị bảng kết quả
    
    # Hiển thị từng cặp nhóm
    print("\nKiểm tra từng cặp nhóm:")
    for comparison in tukey._results_table.data[1:]:
        group1, group2, meandiff, p_adj, lower, upper, reject = comparison
        print(f"Cặp nhóm: {group1} vs {group2}")
        print(f"  - Chênh lệch trung bình: {meandiff}")
        print(f"  - p-value đã điều chỉnh: {p_adj}")
        print(f"  - Khoảng tin cậy: ({lower:.4f}, {upper})")
        print(f"  - Có ý nghĩa thống kê: {'Có' if reject else 'Không'}\n")
    # Tạo DataFrame từ kết quả Tukey HSD
    tukey_results = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    
    # Vẽ biểu đồ hiển thị tất cả các cặp nhóm
    plt.figure(figsize=(12, 8))
    for i, row in tukey_results.iterrows():
        group1 = row['group1']
        group2 = row['group2']
        meandiff = row['meandiff']
        lower = row['lower']
        upper = row['upper']
        reject = row['reject']
        
        # Vẽ đường khoảng tin cậy
        plt.plot([lower, upper], [i, i], color='blue' if reject else 'gray', linewidth=2)
        # Vẽ điểm chênh lệch trung bình
        plt.plot(meandiff, i, 'o', color='red' if reject else 'black')
        # Gắn nhãn cho các cặp nhóm
        plt.text(upper + 0.1, i, f"{group1} vs {group2}", va='center', fontsize=10)
    
    # Tùy chỉnh biểu đồ
    plt.axvline(0, color='red', linestyle='--', linewidth=1)  # Đường tham chiếu tại 0
    plt.title("Tukey HSD Test Results (All Pairs)")
    plt.xlabel("Chênh lệch trung bình")
    plt.ylabel("Cặp nhóm")
    plt.yticks(range(len(tukey_results)), tukey_results['group1'] + " vs " + tukey_results['group2'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
# Gọi hàm kiểm tra Tukey HSD
perform_tukey_hsd(du_lieu_moi, 'Product_Collection', 'Cache')
