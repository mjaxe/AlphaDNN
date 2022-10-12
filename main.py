from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from time import time
import os
import random
from sklearn.model_selection import train_test_split
import math

# PLS DO NOT EXCEED THIS TIME LIMIT
MAXIMIZED_RUNNINGTIME = 1000
# REPRODUCE THE EXP
seed = 123
random.seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

parser = ArgumentParser()
###########################MAGIC HAPPENS HERE##########################
parser.add_argument("--optimizer", default='adam', type=str)
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--hidden_size", default=64, type=int)
parser.add_argument("--scale_factor", default=255, type=float)
###########################MAGIC ENDS HERE##########################

parser.add_argument("--is_pic_vis", action="store_true")
parser.add_argument("--model_output_path", type=str, default="./output")
parser.add_argument("--model_nick_name", type=str, default=None)

args = parser.parse_args()
start_time = time()
# Hyper-parameter tuning
# Custom dataset preprocess

# create the output_adagrad dir if it not exists.
if os.path.exists(args.model_output_path) is False:
    os.mkdir(args.model_output_path)

if args.model_nick_name is None:
    setattr(args, "model_nick_name", f"OPT:{args.optimizer}-E:{args.epochs}-H:{args.hidden_size}-S:{args.scale_factor}")

'''
1. Load the dataset
Please do not change this code block
'''
class_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# check the validity of dataset
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)

# Take the first channel
x_train = x_train[:, :, :, 0]
x_test = x_test[:, :, :, 0]

# split the training dataset into training and validation
# 70% training dataset and 30% validation dataset
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=seed,
                                                      stratify=y_train)

if args.is_pic_vis:
    # Visualize the image
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i][0]])
    plt.show()

'''
2. Dataset Preprocess
'''

# Scale the image
###########################MAGIC HAPPENS HERE##########################
x_train = x_train / args.scale_factor
x_valid = x_valid / args.scale_factor
x_test = x_test / args.scale_factor
###########################MAGIC ENDS HERE##########################

if args.is_pic_vis:
    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

'''
3. Build up Model 
'''
num_labels = 10
model = Sequential()
###########################MAGIC HAPPENS HERE##########################
# Build up a neural network to achieve better performance.
# Hint: Deeper networks (i.e., more hidden layers) and a different activation function may achieve better results.
model.add(Flatten())
model.add(Dense(1024, activation="relu"))  # first layer
model.add(Dense(512, activation="relu"))  # first layer
model.add(Dense(256, activation="relu"))  # first layer
model.add(Dense(128, activation="relu"))  # first layer
###########################MAGIC ENDS HERE##########################
model.add(Dense(num_labels))  # last layer

# Compile Model
model.compile(optimizer=args.optimizer,

              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=args.epochs,
                    batch_size=512)
print(history.history)

training_accuracies = history.history["val_accuracy"]
# Report Results on the test datasets
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest Accuracy: ", test_acc)

end_time = time()
assert end_time - start_time < MAXIMIZED_RUNNINGTIME, "YOU HAVE EXCEED THE TIME LIMIT, PLEASE CONSIDER USE SMALLER EPOCHS and SHAWLLOW LAYERS"
# save the model
model.save(args.model_output_path + "/" + args.model_nick_name)

'''
4. Visualization and Get Confusion Matrix from test dataset 
'''

y_test_predict = np.argmax(model.predict(x_test), axis=1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###########################MAGIC HAPPENS HERE##########################
# Visualize the confusion matrix by matplotlib and sklearn based on y_test_predict and y_test
# Report the precision and recall for 10 different classes
# Hint: check the precision and recall functions from sklearn package or you can implement these function by yourselves.
labels = list(class_names.values())
y_test_predict = list(map(lambda x: class_names[x], y_test_predict))
y_test = list(map(lambda x: class_names[x[0]], y_test))
mtrx = confusion_matrix(y_test, y_test_predict, labels=labels)
prec = precision_score(y_test, y_test_predict, labels=None, pos_label=1, average=None, sample_weight=None,
                       zero_division='warn')
recall = recall_score(y_test, y_test_predict, labels=None, pos_label=1, average=None, sample_weight=None,
                      zero_division='warn')

print(f'val_accuracy: {history.history["val_accuracy"]}')
print(f'precision: {prec}')
print(f'recall: {recall}')

wrong = []
for i, prediction in enumerate(y_test_predict):
    if y_test[i] != prediction:
        wrong.append(i)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

from PIL import Image

options = random.sample(wrong, 3)

f = open("predictions.txt", "w")
f.close()
y_test = list(map(lambda x: class_names[x[0]], y_test))
# saves every wrong image
for ind in options:
    pxls = x_test[ind]
    img = Image.new('RGB', (32, 32), "black")  # Create a new black image
    pixels = img.load()  # Create the pixel map
    for i in range(img.size[0]):  # For every pixel:
        for j in range(img.size[1]):
            pixels[i, j] = (int(pxls[i][j][0]), int(pxls[i][j][1]), int(pxls[i][j][2]))  # Set the colour accordingly
    pred = (ind, f'prediction: {y_test_predict[ind]}', f'actual: {y_test[ind]}')
    f = open("predictions.txt", "a")
    f.write(str(pred) + "\n")
    img.save(f'{ind}.png')
f.close()

fig, ax = plt.subplots(1,2)
im = ax[0].imshow(mtrx)

# Show all ticks and label them with the respective list entries
ax[0].set_xticks(np.arange(len(labels)), labels=labels)
ax[0].set_yticks(np.arange(len(labels)), labels=labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax[0].text(j, i, mtrx[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()

# plt.show()

plt.savefig("Confusion Matrix.png")


fig.tight_layout()

ax[1].plot(range(1, args.epochs + 1), training_accuracies)
plt.savefig("accuracy.png")
###########################MAGIC ENDS HERE##########################
