import pandas as pd
from sklearn.model_selection import train_test_split
# 读取上传的文件
file_path = r'D:\flow_idea2\tensor_mixpre_allnew\tensor_mixpre.csv'
df = pd.read_csv(file_path)

# 更新标签列，只保留数字部分
df['label'] = df['label'].apply(lambda x: x.split('-')[-1])

# 打乱数据
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 分割数据为80%训练集和20%验证集
train_df, valid_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)

# 保存到新的CSV文件
train_file_path = r'D:\flow_idea2\tensor_mixpre_allnew\train_data.csv'
valid_file_path = r'D:\flow_idea2\tensor_mixpre_allnew\valid_data.csv'
train_df.to_csv(train_file_path, index=False)
valid_df.to_csv(valid_file_path, index=False)

print(f'Train data saved to: {train_file_path}')
print(f'Validation data saved to: {valid_file_path}')