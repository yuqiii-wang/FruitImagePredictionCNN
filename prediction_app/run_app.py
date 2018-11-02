import numpy as np
import random
from flask import Flask, render_template, request
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image
import os
import time

app = Flask(__name__)

class str_cmd:
	def __init__(self):
		self.file_path = 'static/'
		self.filename = 'screenshot.jpg'
		self.cp_cmd = "cd && cd Downloads/ && cp screenshot.jpg /Users/yuqi/Desktop/2018/AdvanSigDeepLearning/proj_2/prediction_app/static/screenshot.jpg"
		self.rm_cmd = "cd && cd Downloads/ && rm screenshot.jpg"
		self.rm_old_file_cmd = "rm static/screenshot.jpg"
	
	def change_filename(self, filename):
		# os.system("mv " + self.file_path + self.filename + " " + self.file_path + filename)
		temp_rm_list = self.rm_old_file_cmd.split('/')
		temp_rm_list[-1] = self.filename
		self.rm_old_file_cmd = '/'.join(temp_rm_list)

		self.filename = filename
		temp_cp_list = self.cp_cmd.split('/')
		temp_cp_list[-1] = filename
		self.cp_cmd = '/'.join(temp_cp_list)

label_list = [('apple', 0), ('blueberry', 1), ('pear', 2), ('strawberry', 3), ('avocado', 4), ('pomegranate', 5), ('lemon', 6), ('kiwifruit', 7), ('rockmelon', 8), ('plum', 9), ('cherry', 10), ('banana', 11), ('dragonfruit', 12), ('grape', 13), ('papaya', 14), ('pineapple', 15), ('orange', 16), ('peach', 17)]

str_cmd = str_cmd()

class Vanilla_net(torch.nn.Module):
	# this is for test
	def __init__(self, num_classes=18):
		torch.nn.Module.__init__(self)
		self.fc = torch.nn.Linear(224*224*3, num_classes)
		
	def forward(self, X):
		X = X.view(-1, 224*224*3)
		X = self.fc(X)
		return X 

class Bilinear_CNN(torch.nn.Module):
    def __init__(self, num_classes=18):
        torch.nn.Module.__init__(self)
        
        # only to have some custimizations on alex
        self.alex = torchvision.models.alexnet(pretrained=True).features
        
        self.added_conv = torch.nn.Sequential(
                torch.nn.Conv2d(384, 306, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(306, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                )
        
        self.dense = torch.nn.Sequential(
            torch.nn.Dropout(),
            # size = [10, 256, 6, 6]
            torch.nn.Linear(256 * 256, 6796),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(6796, 996),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(996, 356),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(356, 86),
            torch.nn.ReLU(inplace=True),
        )
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(86, num_classes),
            )
        
    def forward(self, x):
        x = self.alex(x)
        # size = [10, 256, 6, 6]
        x = x.view(x.size(0), 256, 6 * 6)
        x = torch.bmm(x, torch.transpose(x, 1, 2))  # Bilinear
        x = x.view(x.size(0), 256 ** 2)
        x = self.dense(x)
        x = self.clf(x)
        return x


class createDataset(Dataset):
	
	def __init__(self, file_list, lbl, transform=None):
		self.file_list = file_list
		self.lbl = lbl
		self.transform = transform
	
	def __len__(self):
		return len(self.lbl)
	
	def __getitem__(self, idx):
		PIL_img = Image.open(self.file_list[idx])
		sample = {"train_data": PIL_img, "train_labels": self.lbl[idx]}
		if not self.transform is None:
			sample['train_data'] = self.transform(sample['train_data'])
		return sample["train_data"], sample["train_labels"]

def pred(filename, model_file="Vanilla_net.pth"):
	input_crop_size = 224
	transform_comp = transforms.Compose([transforms.CenterCrop((int(input_crop_size*2.1), int(input_crop_size*2.1))), transforms.Resize((input_crop_size, input_crop_size)), transforms.ToTensor()])
	net = Vanilla_net()
	# model_file = 'Bilinear_CNN.pth'
	# net = Bilinear_CNN()

	tensor_data = createDataset([filename], [0], transform=transform_comp)
	tensor_dataloader = DataLoader(tensor_data, batch_size=10, shuffle=True, num_workers=2)
	net.load_state_dict(torch.load(model_file, map_location='cpu'))

	for i, data in enumerate(tensor_dataloader, 0):
	# get the inputs
		inputs, _ = data
		outputs = net(inputs)

	_, predicted = torch.max(outputs.data, 1)
	pred_label = predicted.data.detach().numpy()
	all_pred = outputs.data.detach().numpy()

	def sigmoid(x, derivative=False):
		return x*(1-x) if derivative else 1/(1+np.exp(-x))

	all_pred = np.ravel(all_pred)
	all_pred = sigmoid(all_pred)

	pred_label = label_list[pred_label[0]][0]
	all_pred_dict = {}
	for i in range(len(label_list)):
		all_pred_dict[label_list[i][0]] = all_pred[i]
	return pred_label, all_pred_dict

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(str_cmd.file_path + f.filename)
		pred_label, all_pred = pred(str_cmd.file_path + f.filename)

		data = {"class": str(pred_label)}
		for x in all_pred:
			data[x] = all_pred[x]
		return render_template('result.html', data=data)

	if request.method == 'GET':
		return render_template('upload.html')

@app.route('/show_result', methods = ['GET', 'POST'])
def show_result():
	if request.method == 'GET':
		time.sleep(2) # wait for 1 sec to have all operations done
		pred_label, all_pred = pred(str_cmd.file_path + str_cmd.filename)

		data = {"class": str(pred_label)}
		for x in all_pred:
			data[x] = all_pred[x]
		user_img_file = str_cmd.file_path + str_cmd.filename
		return render_template('result.html', data=data, user_img_file=user_img_file)

@app.route('/webcam', methods = ['GET', 'POST'])
def webcam_capture():
	if request.method == 'GET':
		return render_template('webcam_capt.html')

@app.route('/auto_img_upload', methods = ['GET'])
def auto_img_upload():
	if request.method == 'GET':
		time.sleep(1) # wait for 1 sec to have all downloading operations done
		ticks = time.time()
		str_cmd.change_filename(str(round(ticks)) + '.jpg')
		os.system(str_cmd.cp_cmd)
		os.system(str_cmd.rm_cmd)
		return "200"
		
if __name__ == '__main__':
   app.run(debug = True)