import os
import pandas as pd
import numpy as np
import json
#from google.colab import files
import io
#from keras import layers
#from keras import models
#from keras import applications
#from keras import optimizers
#import keras
import tensorflow as tf
def predict(path,f):

    #path='/content/drive/My Drive/Training/v10.json'
    #path='/content/1'
    json_files=[pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    print('Found: ',len(json_files),'json keypoint frame files')
    a=json.load(open(path+'/'+json_files[0]))
    b=a['people']
    if not b:
        print('true')
        s=np.zeros(50)
        train=pd.DataFrame([s])
    else:
        q=b[0]['pose_keypoints_2d']
        s=[p for p in q if p>1]
        train=pd.DataFrame([s])
    #q=b[0]['pose_keypoints_2d']
    #s=[p for p in q if p>1]
    #train=pd.DataFrame([s])
    for j in range(1,len(json_files)):
        a=json.load(open(path+'/'+json_files[j]))
        b=a['people']
        if len(b)>0:
            if not b:
                s=np.zeros(50)
                s=pd.DataFrame([s])
                train=train.append(s)
            else:
                q=b[0]['pose_keypoints_2d']
                s=[p for p in q if p>1]
                s=pd.DataFrame([s])
                train=train.append(s)
        else:
            continue

    train=train.fillna(method='ffill')
    train=train.fillna(0)
    y,x=train.shape
    s=np.zeros(y)
    s=pd.DataFrame(s)

    for i in range(50):
        y,x=train.shape
        if x!=50:
            train[x]=s
            x=x+1
        if y<90:
            train=train.append(train.iloc[y-1])

    for i in range(50):
        train.iloc[:,i]=train.iloc[:,i]/train.iloc[:,i].max()

    train=train.fillna(method='ffill')
    train=train.fillna(0)
    train.head()
    x,y=train.shape
    train=train.iloc[:45*int(x/45),:30]
    train=train.values
    train=train.reshape(int(x/45),45,30)
    #model.load_weights('/content/weights11.221-0.96.hdf5')
    model=tf.keras.models.load_model('wtsfinalw3.h5')
    y=model.predict(train)
    y=y.flatten()
    len(y)
    model=tf.keras.models.load_model('wtsconvlstm.h5')
    y1=model.predict([train,train])
    y1=y1.flatten()
    y2=y1*0.5+0.5*y
    #import statistics
    #t1=[]
    #for i in range(60):
    #    t1.append(statistics.mean(y2[i*30:(i+1)*30]))

    import matplotlib.pyplot as plt 
    plt.ylim(-0.5,1.5)
    plt.plot(y2)
    ax=plt.axes()
    ax.set_xticks(np.arange(0,len(y),30))
    ax.set_xticklabels(np.arange(0,int(len(y)/30),1))
    plt.xlabel('time')
    plt.ylabel('action')
    
    plt.savefig('output graphs/'+str(path[5:11])+'_Part6.jpg')
    plt.show()
    #plt.xticks(np.arange(0,60,1)
    s=[]
    #for i in range(len(t1)):
    #    s.append([i,float(t1[i])])
    #q=json.dumps(s)
    #with open('timelabels json/929004194 Timelabelvideo'+str(f)+'.json','w') as outfile:
    #    outfile.write(q)

path='test/7100_1.json'
print('4')
predict(path,4)
path='test/7200_1.json'
print('5')
predict(path,5)
path='test/7200_4.json'
print('8')
predict(path,8)
path='test/7300_1.json'
print('9')
predict(path,9)
path='test/7400_4.json'
print('10')
predict(path,10)
path='test/7500_3.json'
print('11')
predict(path,11)