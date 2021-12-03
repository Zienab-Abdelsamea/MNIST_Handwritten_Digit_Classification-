#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ## -load mnist data and see an example of it.

# In[19]:


#loading mnist data
mnist = tf.keras.datasets.mnist 
#seperate data into traning and testing
# the x_train data is the "features." In this case, the features are the images of digits 0-9.
#The y_train is the label (is it a 0,1,2,3,4,5,6,7,8 or a 9?)
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[20]:


#to see an example of our data
print(x_train[0]) 


# ## -Visualizing the first element of our data
# 

# In[21]:


#pixcels 28*28 of first data pic
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])


# ## -Normalize data
# 
# 
# 
# 

# In[22]:


#normalizing our data to range from 0 to 1 instead of 0 to 255 
#to simplize the task to the nural network
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[23]:


#have a look again
print(x_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# ## -Model Building and Validation
# <li>Initialize a sequential model</li>
# <li>define a sequential model</li>
# <li>add 2 convolutional layers</li>
# <li>no of filters: 32</li>
# <li>kernel size: 3x3</li>
# <li>activation: "relu"</li>
# <li>input shape: (28, 28, 1) for first layer</li>
# <li>flatten the data</li>
# <li>add Flatten later</li>
# <li>flatten layers flatten 2D arrays to 1D array before building the fully connected layers</li>
# <li>add 2 dense layers</li>
# <li>number of neurons in first layer: 128</li>
# <li>number of neurons in last layer: number of classes</li>
# <li>activation function in first layer: relu</li>
# <li>activation function in last layer: softmax</li>
# <li>we may experiment with any number of neurons for the first Dense layer; however, the final Dense layer must have</li>
# <li>neurons equal to the number of output classes</li>

# In[24]:


#let's build our model! we will use a Sequential model
#It just means things are going to go in direct order.
#A feed forward model. No going backwards...for now
model = tf.keras.models.Sequential()


# In[25]:


#now we have to flatten the pic from 28*28 to 1*784 to be able to feed this features (pixcels) to the input layer of our neural network
model.add(tf.keras.layers.Flatten())


# ### -Creat Nurel Network

# In[26]:


#creating a hidden layer of 128 node with activation function relu
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
#adding another layer to our NN of 128 node and relu AF
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#adding the output layer consist of 10 nodes represent the 10 digits with softmax AF
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
#now our model is DONE 


# In[27]:


#This is where we pass the settings for actually optimizing/training the model we've defined
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### -Fit the model
# 
# 

# In[28]:


#Now, we fit
model.fit(x_train, y_train, epochs=3)


# ### -Evaluate our model

# In[29]:


#lets evaluate our model 
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
#It's going to be very likely your accuracy out of sample is a bit worse,
#same with loss.In fact, it should be a red flag if it's identical, or better.


# In[30]:


predictions=model.predict(x_test)
print (predictions[0])


# In[31]:


# numpy argmax function return the index of the max value of the array
#in our case the indix represent the predictied number
print(np.argmax(predictions[0]))


# In[32]:


#lets plot it up
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()


# In[33]:


# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = model.predict(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='pink')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions[i]))


# In[34]:


model.summary()


# In[ ]:




