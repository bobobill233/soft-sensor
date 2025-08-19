import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_concatenate_images(csv_file, save_folder1, output_csv):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    gasf_paths = df['gasf_path'].tolist()
    mtf_paths = df['mtf_path'].tolist()
    labels = df['label'].tolist()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 确保图像尺寸一致
        # transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
        # transforms.CenterCrop(192),  # 从中心开始裁剪
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        # transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_records = []

    for gasf_path, mtf_path, label in zip(gasf_paths, mtf_paths, labels):
        gasf_img = Image.open(gasf_path).convert('RGB')
        mtf_img = Image.open(mtf_path).convert('RGB')

        transform_gasf = transform(gasf_img)
        transform_mtf = transform(mtf_img)

        gasf_tensor_path = os.path.join(save_folder1, f'{label}_gasf.pt')
        mtf_tensor_path = os.path.join(save_folder1, f'{label}_mtf.pt')

        torch.save(transform_gasf, gasf_tensor_path)
        torch.save(transform_mtf, mtf_tensor_path)

        data_records.append({
            'gasf_tensor_path': gasf_tensor_path,
            'mtf_tensor_path': mtf_tensor_path,
            'label': label
        })

    output_df = pd.DataFrame(data_records)
    output_df.to_csv(output_csv, index=False)
    print("张量保存完成并已更新CSV文件！")

csv_file = r'D:\flow_idea2\picture_pathsnew.csv'
save_folder1 = r'D:\flow_idea2\tensor_mixpre_allnew'

if not os.path.exists(save_folder1):
    os.makedirs(save_folder1)

output_csv = save_folder1 + r'\tensor_mixpre.csv'
load_and_concatenate_images(csv_file, save_folder1, output_csv)



# # 读取上传的文件
# file_path = r'D:\flow_idea2\tensor_mixpre_all\tensor_mixpre1.csv'
# df = pd.read_csv(file_path)
#
# # 更新标签列，只保留数字部分
# df['label'] = df['label'].apply(lambda x: x.split('-')[-1])
#
# # 打乱数据
# df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # 分割数据为80%训练集和20%验证集
# train_df, valid_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)
#
# # 保存到新的CSV文件
# train_file_path = r'D:\flow_idea2\tensor_mixpre_all1\train_data.csv'
# valid_file_path = r'D:\flow_idea2\tensor_mixpre_all1\valid_data.csv'
# train_df.to_csv(train_file_path, index=False)
# valid_df.to_csv(valid_file_path, index=False)
#
# print(f'Train data saved to: {train_file_path}')
# print(f'Validation data saved to: {valid_file_path}')
