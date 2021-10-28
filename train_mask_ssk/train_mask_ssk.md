```python
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"./dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

# complete folder of dataset is enter into data and labels(it has category) list
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels
# As category having 2 fields with mask, without mask we need to convert them into number(0 and 1) so called LabelBinarizer()
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# converting both data and labels list to array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# splitting into train and test data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# imagenet is pretrained model specifically for images... as we are using images those predefined weights will be initialize for us 
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3))) # i.e 224x224 size and 3 stands for RGB

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)  # relu is basically goto activation function for non-linear cases...(as images use relu)
headModel = Dropout(0.5)(headModel)  # dropout to avoid overfitting
headModel = Dense(2, activation="softmax")(headModel) # softmax because there probabilty values based on 0 and 1
# dense 2 because 1 is for with mask and other without mask

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

```

    [INFO] loading images...
    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
    [INFO] compiling model...
    [INFO] training head...
    Epoch 1/20
    34/34 [==============================] - 90s 3s/step - loss: 0.5466 - accuracy: 0.7481 - val_loss: 0.1370 - val_accuracy: 0.9601
    Epoch 2/20
    34/34 [==============================] - 80s 2s/step - loss: 0.1366 - accuracy: 0.9532 - val_loss: 0.0717 - val_accuracy: 0.9819
    Epoch 3/20
    34/34 [==============================] - 92s 3s/step - loss: 0.0769 - accuracy: 0.9719 - val_loss: 0.0503 - val_accuracy: 0.9855
    Epoch 4/20
    34/34 [==============================] - 92s 3s/step - loss: 0.0496 - accuracy: 0.9878 - val_loss: 0.0391 - val_accuracy: 0.9928
    Epoch 5/20
    34/34 [==============================] - 83s 2s/step - loss: 0.0399 - accuracy: 0.9888 - val_loss: 0.0336 - val_accuracy: 0.9891
    Epoch 6/20
    34/34 [==============================] - 89s 3s/step - loss: 0.0431 - accuracy: 0.9888 - val_loss: 0.0314 - val_accuracy: 0.9928
    Epoch 7/20
    34/34 [==============================] - 102s 3s/step - loss: 0.0281 - accuracy: 0.9944 - val_loss: 0.0279 - val_accuracy: 0.9891
    Epoch 8/20
    34/34 [==============================] - 97s 3s/step - loss: 0.0337 - accuracy: 0.9869 - val_loss: 0.0276 - val_accuracy: 0.9928
    Epoch 9/20
    34/34 [==============================] - 100s 3s/step - loss: 0.0228 - accuracy: 0.9934 - val_loss: 0.0250 - val_accuracy: 0.9855
    Epoch 10/20
    34/34 [==============================] - 94s 3s/step - loss: 0.0268 - accuracy: 0.9906 - val_loss: 0.0258 - val_accuracy: 0.9928
    Epoch 11/20
    34/34 [==============================] - 91s 3s/step - loss: 0.0238 - accuracy: 0.9925 - val_loss: 0.0220 - val_accuracy: 0.9928
    Epoch 12/20
    34/34 [==============================] - 88s 3s/step - loss: 0.0257 - accuracy: 0.9944 - val_loss: 0.0202 - val_accuracy: 0.9891
    Epoch 13/20
    34/34 [==============================] - 84s 2s/step - loss: 0.0141 - accuracy: 0.9953 - val_loss: 0.0192 - val_accuracy: 0.9891
    Epoch 14/20
    34/34 [==============================] - 86s 3s/step - loss: 0.0132 - accuracy: 0.9963 - val_loss: 0.0181 - val_accuracy: 0.9891
    Epoch 15/20
    34/34 [==============================] - 78s 2s/step - loss: 0.0116 - accuracy: 0.9972 - val_loss: 0.0163 - val_accuracy: 0.9891
    Epoch 16/20
    34/34 [==============================] - 81s 2s/step - loss: 0.0133 - accuracy: 0.9953 - val_loss: 0.0154 - val_accuracy: 0.9964
    Epoch 17/20
    34/34 [==============================] - 79s 2s/step - loss: 0.0127 - accuracy: 0.9972 - val_loss: 0.0156 - val_accuracy: 0.9891
    Epoch 18/20
    34/34 [==============================] - 86s 3s/step - loss: 0.0169 - accuracy: 0.9954 - val_loss: 0.0150 - val_accuracy: 0.9928
    Epoch 19/20
    34/34 [==============================] - 99s 3s/step - loss: 0.0084 - accuracy: 1.0000 - val_loss: 0.0146 - val_accuracy: 0.9928
    Epoch 20/20
    34/34 [==============================] - 95s 3s/step - loss: 0.0129 - accuracy: 0.9944 - val_loss: 0.0133 - val_accuracy: 0.9928
    [INFO] evaluating network...
                  precision    recall  f1-score   support
    
       with_mask       0.99      0.99      0.99       138
    without_mask       0.99      0.99      0.99       138
    
        accuracy                           0.99       276
       macro avg       0.99      0.99      0.99       276
    weighted avg       0.99      0.99      0.99       276
    
    [INFO] saving mask detector model...
    


![png](output_0_1.png)



```python

```
