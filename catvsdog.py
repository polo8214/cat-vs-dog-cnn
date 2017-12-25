#%%

import cv2
import numpy as np
import os 
from random import shuffle
from tqdm import tqdm

train_dir = '/home/andy/catanddog_dataset/train/'
test_dir = '/home/andy/catanddog_dataset/test/'
img_size = 100
LR = 0.001
count = 0

MODEL_NAME = 'dogvscats-{}-{}.model'.format(LR, '6conv-basic_100')

#%%

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]
        
#%%
        
def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data_100.npy', train_data)
    return train_data
    
#%%

def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        test_data.append([np.array(img), img_num])
        
    np.save('test_data_100.npy', test_data)
    return test_data
    
#%%

training_data = create_train_data()
# If you have already created the dataset:
#training_data = np.load('train_data.npy')
        
#%%        #CNN
        
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')    #input tensor , number of filter , filter size , activation func
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')


#%%

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
   
#%%

train = training_data[:-500]    #all after 500
test = training_data[-500:]     #the first 500

X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
test_y = [i[1] for i in test]

#%%  #Run Training

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    
#%%     #save the model
model.save(MODEL_NAME)    

#%%     #output 

import matplotlib.pyplot as plt
#if you don't have this file yet
#testing_data = process_test_data()
#if you already have it 
testing_data = np.load('test_data_100.npy')

fig = plt.figure()


for num, data in enumerate(testing_data[count:count+4]):
        #cat[0,1]
        #dog[1,0]
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(2,2,num+1)
    orig = img_data
    data = img_data.reshape(img_size,img_size,1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) ==1: str_label = 'dog'
    else: str_label = 'cat'
    
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show

count = count + 4
