import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist #Get the data

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#Create Neural Network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3) #Train model 1 time.

#Show the RGB codes
#print(x_train[0])

#Show the loss and accuracy.
val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)

model.save('epic_num_reader.model.h5')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])

#Convert to understandable form.
print("The number is : ",np.argmax(predictions[10])) 
#If you want change the test and prediction numbers

#Show the image
plt.imshow(x_test[10]) 
plt.show()
