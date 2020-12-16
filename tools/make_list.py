import os
import sys
from sklearn.model_selection import train_test_split

base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)  # 设置项目根目录

image_path = os.path.abspath(os.path.join(base_dir, "dataset/image_crnn/"))
train_path = os.path.abspath(os.path.join(base_dir, "dataset/train.txt"))
eval_path = os.path.abspath(os.path.join(base_dir, "dataset/eval.txt"))

image_list = os.listdir(image_path)     # 文件列表
x_train, x_eval = train_test_split(image_list, test_size=0.1, random_state=0)

fp_train = open(train_path, "w", encoding="utf-8")
fp_eval = open(eval_path, "w", encoding="utf-8")

for image_single_name in x_train:
    image_single_path = os.path.abspath(os.path.join(image_path, image_single_name))
    fp_train.write(image_single_path)
    fp_train.write("\n")
fp_train.close()

for image_single_name in x_eval:
    image_single_path = os.path.abspath(os.path.join(image_path, image_single_name))
    fp_eval.write(image_single_path)
    fp_eval.write("\n")
fp_eval.close()


