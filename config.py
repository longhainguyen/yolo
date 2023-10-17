# kích thước grid system 
cell_size = 8
# số boundary box cần dự đoán mỗi ô vuông
box_per_cell = 2
# kích thước ảnh đầu vào
img_size_H = 480
img_size_W = 640
#kích thước ảnh đầu ra
out_H = 8
out_W = 10
# số loại nhãn
classes = {'Platelets':0, 'RBC':1,  'WBC':2}
nclass = len(classes)

box_scale = 5.0
noobject_scale = 0.5
batch_size = 4
# số lần huấn luyện
epochs = 100
# learning của chúng ta
lr = 1e-3
NUM_WORKERS = 2
PIN_MEMORY = True
#data dir
DATA_LABLE_DIR = "./data/labels/train/"
DATA_IMG_DIR = "./data/images/train/"