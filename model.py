#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import seed
seed(1337)
import tensorflow
tensorflow.random.set_seed(42)


# In[ ]:





# # DATA

# In[2]:


train_data_dir = "DATA00/train"
test_data_dir = "DATA00/test"
val_data_dir = "DATA00/validation"
category_names = sorted(os.listdir('DATA00/train'))
nb_categories = len(category_names)
img_pr_cat = []


# In[3]:


for category in category_names:
    folder = 'DATA00/train' + '/' + category
    img_pr_cat.append(len(os.listdir(folder)))
sns.barplot(y=category_names, x=img_pr_cat).set_title("Number of training images per category:")


# In[4]:


category_names = sorted(os.listdir('DATA00/test'))
nb_categories = len(category_names)
img_pr_cat = []
for category in category_names:
    folder = 'DATA00/test' + '/' + category
    img_pr_cat.append(len(os.listdir(folder)))
sns.barplot(y=category_names, x=img_pr_cat).set_title("Number of test images per category:")


# # image quality

# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
for subdir, dirs, files in os.walk('DATA00/train'):
    for file in files:
        img_file = subdir + '/' + file
        image = load_img(img_file)
        plt.figure()
        plt.title(subdir)
        plt.imshow(image)
        break


# # Pre-Processing 

# In[6]:


img_height, img_width = 224,224


# In[7]:


#Number of images to load at each iteration
batch_size = 45


#  rescaling
train_datagen =  ImageDataGenerator(
    rescale=1./255
)
test_datagen =  ImageDataGenerator(
    rescale=1./255
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)



# these are generators for train/test data that will read pictures #found in the defined subfolders of 'data/'
print('Total number of images for "training":')
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

print('Total number of images for "test":')
val_generator = val_datagen.flow_from_directory(
val_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical",
shuffle=False)

print('Total number of images for "test":')
test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical",
shuffle=False)


# In[8]:


import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, Model, optimizers 
from tensorflow.keras.optimizers import Adam


# # downloading VGG pretrained model

# In[9]:


base_model =  vgg16.VGG16(weights='imagenet', include_top=False, pooling='max', input_shape = (img_width, img_height, 3))


# In[10]:


type(base_model)


# In[11]:


from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model

def build_finetune_model(base_model, dropout, fc_layers, nb_categories):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(nb_categories, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

FC_LAYERS = [1024, 1024]
dropout = 0.5

model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      nb_categories=len(category_names))


# In[12]:


model.summary()


# # ADDING LAYERS TO PRETRAINED MODEL 

# In[ ]:





# ## add our new output layer, consisting of only 2 nodes that correspond to NSFW and SFW. This output layer will be the only trainable layer in the model.

# In[ ]:





# In[13]:


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


checkpoint = ModelCheckpoint("vgg_weights.h5", monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto',period=1)


# In[15]:


#Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=
         train_generator.samples/train_generator.batch_size,
      epochs=5,
      validation_data=val_generator, 
      validation_steps=
         val_generator.samples/val_generator.batch_size,
      verbose=1,workers=8,
callbacks=[checkpoint])


# In[ ]:





# 
# # Save the Model Weights
# model.save_weights('model_100_eopchs_adam_20191030_01.h5')
# 
# # Save the Model to JSON
# model_json = model.to_json()
# with open('model_adam_20191030_01.json', 'w') as json_file:
#     json_file.write(model_json)
#     
# print('Model saved to the disk.')

# In[ ]:





# # PLOTTING OF MODEL RESULTS

# In[16]:


# Utility function for plotting of the model results
def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


# Run the function to illustrate accuracy and loss
visualize_results(history)


# In[36]:


model.save('final_model.h5')


# In[52]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('Desktop/card.jpeg')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict(img)
	print(result[0])

run_example()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # visualization of the errors

# # Utility function for obtaining of the errors 
# def obtain_errors(val_generator, predictions):
#     # Get the filenames from the generator
#     fnames = test_generator.filenames
# 
#     # Get the ground truth from generator
#     ground_truth = test_generator.classes
# 
#     # Get the dictionary of classes
#     label2index = test_generator.class_indices
# 
#     # Obtain the list of the classes
#     idx2label = list(label2index.keys())
#     print("The list of classes: ", idx2label)
# 
#     # Get the class index
#     predicted_classes = np.argmax(predictions, axis=1)
# 
#     errors = np.where(predicted_classes != ground_truth)[0]
#     print("Number of errors = {}/{}".format(len(errors),test_generator.samples))
#     
#     return idx2label, errors, fnames
# 

# # Utility function for visualization of the errors
# def show_errors(idx2label, errors, predictions, fnames):
#     # Show the errors
#     for i in range(len(errors)):
#         pred_class = np.argmax(predictions[errors[i]])
#         pred_label = idx2label[pred_class]
# 
#         title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#             fnames[errors[i]].split('/')[0],
#             pred_label,
#             predictions[errors[i]][pred_class])
# 
#         original = load_img('{}/{}'.format(test_data_dir,fnames[errors[i]]))
#         plt.figure(figsize=[7,7])
#         plt.axis('off')
#         plt.title(title)
#         plt.imshow(original)
#         plt.show()

# # Get the predictions from the model using the generator
# predictions = model.predict(test_generator, steps=test_generator.samples/test_generator.batch_size,verbose=1)
# 
# # Run the function to get the list of classes and errors
# idx2label, errors, fnames = obtain_errors(test_generator, predictions)
# 
# # Run the function to illustrate the error cases
# show_errors(idx2label, errors, predictions, fnames)

# loss, accuracy = finetune_model.evaluate(test_generator)
# print('Test accuracy :', accuracy)

# #Save the model
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# In[45]:


model.save("model_num.h5")

model = load_model('model_num.h5')


# In[44]:





# # TESTING IMAGES BY URL

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
# 
# img = tensorflow.keras.preprocessing.image.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
# 
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# 
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(category_names[np.argmax(score)], 100 * np.max(score))
# )
# 

# In[ ]:





# ## Evaluate on validation set
# 

# # PLOT CONFUSION MATRIX 

# In[ ]:





# In[ ]:





# In[ ]:





# 
# 

# In[ ]:





# # SAVE THE MODEL

# In[ ]:




