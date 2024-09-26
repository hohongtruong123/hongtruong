#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #đọc dữ liệu 
import numpy as np #xử lý dữ liệu
import matplotlib.pyplot as plt #vẽ biểu đồ
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler#chuẩn hóa dữ liệu 
import tensorflow as tf
from tensorflow.keras.models import load_model #tải mô hình
from keras.callbacks import ModelCheckpoint #lưu lại huấn luyện tốt nhất

#các lớp để xây dựng mô hình 
from tensorflow.keras.models import Sequential #đầu vào;/
from tensorflow.keras.layers import LSTM #học phụ thuộc
from tensorflow.keras.layers import Dropout #tránh học tủ 
from tensorflow.keras.layers import Dense #đầu ra 

#kiểm tra độ chính xác của mô hình 
from sklearn.metrics import r2_score #đo mức độ phù hợp 
from sklearn.metrics import mean_absolute_error #đo sai số tuyệt đối trung bình 
from sklearn.metrics import mean_absolute_percentage_error #đo phần trăm sai số tuyệt đối trung bình 


# In[2]:


TCB=pd.read_excel("C:\DATA\TCB.xlsx")
TCB.sort_values(by='Date', ascending=True, inplace=True)
TCB


# In[3]:


data=pd.DataFrame(TCB, columns=['Date', 'Price'])
data


# In[4]:


data.set_index('Date', inplace=True)


# In[5]:


data


# In[6]:


plt.figure(figsize=(10,5))
plt.plot(data['Price'], label='giá thực tế', color='red')#lập biểu đồ 
plt.title('Biểu đồ giá cổ phiếu') #đặt tên biểu đồ 
plt.xlabel('thời gian')#đặt tên hàm x
plt.ylabel('giá đóng cửa(VND)') #đặt tên cột y
plt.legend()
plt.show()


# In[7]:


#chia tập dữ liệu 
TCB_F=data.values
train_data=TCB_F[:800]
test_data=TCB_F[800:]


# In[8]:


TCB_F


# In[9]:


#chuẩn hóa dữ liệu
sc=MinMaxScaler(feature_range=(0,1))#MinMaxScaler sẽ chuẩn hóa dữ liệu với giá trị nhỏ nhất 0 và lớn nhất là 1 các giá trị còn lại thí sẽ phân bổ trong khoảng 0 đến 1
sc_train=sc.fit_transform(TCB_F)


# In[10]:


#Mô hình thì sẽ dựa trên 50 ngày cơ sở để dự báo giá cho 50 ngày tiếp theo
#tạo các vòng lặp giá trị
x_train, y_train=[], []
for i in range(50, len(train_data)):
    x_train.append(sc_train[i-50:i,0])
    y_train.append(sc_train[i,0])


# In[11]:


x_train #là bao gồm cái mãng danh sách và mỗi mãng là bao gồm 50 giá đóng cửa lên tục


# In[12]:


y_train #là danh sách giá đóng cửa ngày hôm sau tương ứng với mỗi mãng của x_train


# In[13]:


#xếp dữ liệu thành 1 mảng 
x_train=np.array(x_train)
y_train=np.array(y_train)

#xếp lại dữ liệu thành mảng 1 chiều 
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
y_train=np.reshape(y_train, (y_train.shape[0],1))


# In[14]:


#Xây dựng mô hình và huẩn luyện mô hình 
#xây dựng mô hình (gồm 5 lớp 1 lớp đầu vào Sequential, và 2 lớp LSTM, và một lớp Dropout và một lớp output)
model=Sequential() #tạo lớp mảng cho dữ liệu đầu vào 
model.add(LSTM(units=128, input_shape=(x_train.shape[1],1), return_sequences=True)) #kết nối với đầu vào nên ta phải miêu tả thông tin của đầu vào là input_shape=(x_train)
model.add(LSTM(units=64))
model.add(Dropout(0,5)) #giúp bỏ qua một số đơn vị ngẫu nhiên trách học tủ 
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam') #loss đo sai số tuyệt đối trung bình, sử dụng trình đối ưu hóa adam   


# In[15]:


#huấn luyện mô hình 
save_model="save_model.keras" #mô hình sao khi huấn luyện sẽ được lưu lại dưới dạng file "save_model.hdf5" 
best_model=ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto') #tìm ra mô hình huấn luyện tốt nhất để lưu vào file phía trên, tham số quan sát hàm sai số tuyệt đối trung bình, save_best_only=True là có nghĩa mô hình sao khi huấn luyện chỉ lưu lại một mô hình tốt nhất 
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[best_model]) #tiến hành huấn luyện x_train và y _train là với 100 lần lập(epochs)


# In[16]:


#dữ liệu train
y_train=sc.inverse_transform(y_train) #giá thực 
final_model=load_model('save_model.keras')#tiến hành tải lên lại mô hình tối ưu vừa lưu 
y_train_predict=final_model.predict(x_train)
y_train_predict=sc.inverse_transform(y_train_predict) #giá dự đoán #định giá lại giá trị về giá trị góc bằng sc.inverse


# In[17]:


#xử lý dữ liệu test với việc sử lý dữ liệu cho tập test ta xử lý tư tượng như tập train
test=data[len(train_data)-50:].values
test=test.reshape(-1,1)
sc_test=sc.transform(test)

x_test=[]
for i in range(50,test.shape[0]):
    x_test.append(sc_test[i-50:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#dữ liệu test
y_test=TCB_F[800:] #giá thực
y_test_predict=final_model.predict(x_test)
y_test_predict=sc.inverse_transform(y_test_predict) #giá dự đoán


# In[21]:


#Đánh giá độ chính xác của mô hình ta có biểu đồ dự báo ở hai tập 
#lập biểu đồ so sánh 
train_data1=data[50:800]
test_data1=data[800:] 

plt.figure(figsize=(24,8))
plt.plot(data, label='giá thực tế', color='red') #đường giá thực 
train_data1['giá dự đoán']=y_train_predict #thêm dữ liệu 
plt.plot(train_data1['giá dự đoán'], label='giá dự đoán train', color='green') #đường dự báo giá train
test_data1['giá dự đoán']=y_test_predict #thêm dữ liệu
plt.plot(test_data1['giá dự đoán'], label='giá dự đoán test', color='blue') #đường giá dự báo test
plt.title('So sánh giá dự đoán và giá thực tế') #đặt tên 
plt.xlabel('thời gian') #đặt tên hàm x
plt.ylabel('giá đóng cửa VND') #đặt tên hàm y
plt.legend() #chú thích 
plt.show()


# In[22]:


#r2 
print('Độ phù hợp tập train:', r2_score(y_train, y_train_predict))
#mae
print('Sai số tuyệt đối trung bình tập train:', mean_absolute_error(y_train, y_train_predict))
#mape
print('Phần trăm sai số tuyệt đối trung bình tập train:', mean_absolute_percentage_error(y_train, y_train_predict))


# In[ ]:




