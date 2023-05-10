# Project :  Image Classification using Dogs vs Cats Kaggle dataset

## Introduction: Image Classification - is it a cat or a dog?
The ultimate goal of this project is to create a system that can detect cats and dogs. Dataset used for this project is dogs vs cats dataset from kaggle. 

## About the dataset
The dogs vs cats dataset was first introduced for a Kaggle competition in 2013. The link for the dataset is [here](https://www.kaggle.com/competitions/dogs-vs-cats/data). The dataset is comprised of photos of dogs and cats provided as a subset of photos from a much larger dataset of 3 million manually annotated photos. The dataset was developed as a partnership between Petfinder.com and Microsoft.
- Training data :  25000 samples
- Test data : 12500 samples

## Dataset Exploration and Preprocessing

### Preprocessing of training data
We processed the trainig dataset into a pandas dataframe with two columns names filenames and label, where filename indicates the filename of the image while the label value would be 1 if it is a dog, else o(in case of cat).
```python
import pandas as pd

filenames = os.listdir('data/train')
labels = []
for filename in filenames:
    label = filename.split('.')[0]
    if label == 'dog':
        labels.append(1)
    else:
        labels.append(0)
labels = [str(i) for i in labels]
df_train = pd.DataFrame({'filename': filenames, 'label': labels})
df_train
```
<img width="264" alt="Screenshot 2023-05-10 at 2 39 39 PM" src="https://github.com/srilakshmi-thota/project-3/assets/37259010/3284fd54-5801-4579-9a59-2010ec0e97f1">

### Training data distribution
Training dataset consists of 2500 images of dogs and cats with about 12500 images of cats and 12500 images of dogs. The training dataset is well balanced and has no bias towards either of the classes. Below is the bar plot displaying the distribution of the training dataset where 1 represents dog and 0 represents cat.

<img width="660" alt="Screenshot 2023-05-10 at 2 35 00 PM" src="https://github.com/srilakshmi-thota/project-3/assets/37259010/335f69f0-0639-409b-8e7d-22a33f606040">

### Splitting of training and validation dataset
We had split training dataset into train and validation data in the ratio 80:20. We load and preprocess the training data using the ImageDataGenerator class from Keras. We specify the directory where the training images are stored, the target image size, the batch size, and the class mode.
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# splitting to train & val
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=100)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(train_df, 'data/train', x_col='filename', y_col='label', target_size=(224,224), class_mode='binary', batch_size=64, shuffle = False)

val_datagen  = ImageDataGenerator(rescale=1./255.)
val_generator = val_datagen.flow_from_dataframe(val_df, 'data/train', x_col='filename', y_col='label', target_size=(224,224), class_mode='binary', batch_size=64, shuffle = False)
```
Below image displays few samples from one of the batch from train_generator
![image](https://github.com/srilakshmi-thota/project-3/assets/37259010/d110e8ea-65a9-4be7-bda4-3d8fc625d1d8)

## Util methods
Written a utility method to plot the accuracy and loss curves from the history of training the models. And another for plotting the confusion matrix and printing the classification report of the model by evaluating it on the validation generator.The code snippet for the same is below.
```python
def summarize_history(history, filename):
    plt.figure(figsize=(12,10), dpi=100)
    plt.subplot(211)
    plt.title('Binary Cross Entropy Loss using'+ filename)
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend(['train', 'val'], loc='best')

    plt.subplot(212)
    plt.title('Classification Accuracy using'+filename)
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    # save plot to file
    plt.legend(['train', 'val'], loc='best')
    plt.savefig(filename + 'history_plot.png')
    plt.show()
    return
import seaborn as sns
def show_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(8,6), dpi=100)
    sns.set(font_scale = 1.1)
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', )

    ax.set_xlabel("Predicted Classification", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Cat', 'Dog'])

    ax.set_ylabel("Actual Classification", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Cat', 'Dog'])

    ax.set_title("Confusion Matrix for " + model_name + " Model", fontsize=14, pad=20)
    plt.show()
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_evaluation import plot
def evaluate_model(model, val_generator, model_name):
    threshold = 0.5
    predict = model.predict(val_generator)
    y_pred = np.where(predict > threshold, 1,0)
    conf_matrix = confusion_matrix(val_generator.classes, y_pred)
    show_confusion_matrix(conf_matrix,model_name)
    print('\n Classification Report')
    target_names = ['Cats', 'Dogs']
    report = classification_report(val_generator.classes, y_pred, target_names=target_names)
    print(report)
```

## Models Used

### Baseline CNN Model
- It is a convolutional neural network (CNN) model with four convolutional layers, followed by max pooling layers and two fully connected layers. 
- The first convolutional layer has 32 filters of size 3x3 and uses the ReLU activation function. It takes input with shape (224, 224, 3), which corresponds to 224x224 color images with 3 channels.
- The first max pooling layer has a pool size of 2x2, which downsamples the feature maps by taking the maximum value in each 2x2 region.
- The second convolutional layer has 64 filters of size 3x3 and also uses the ReLU activation function.
- The second max pooling layer again has a pool size of 2x2.
- The third convolutional layer has 128 filters of size 3x3 and also uses the ReLU activation function.
- The third max pooling layer again has a pool size of 2x2.
- The fourth convolutional layer has 128 filters of size 3x3 and also uses the ReLU activation function.
- The fourth max pooling layer again has a pool size of 2x2.
- The flattened output from the last max pooling layer is then passed to two fully connected (dense) layers. The first dense layer has 512 units and uses the ReLU activation function. The second dense layer has a single unit and uses the sigmoid activation function, which outputs a value between 0 and 1, representing the predicted probability of the input image being a dog (1) or a cat (0).
<img width="545" alt="Screenshot 2023-05-10 at 2 56 53 PM" src="https://github.com/srilakshmi-thota/project-3/assets/37259010/ff8f543f-061a-41db-bfb7-88b29102ad97">
