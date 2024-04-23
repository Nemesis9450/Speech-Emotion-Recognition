# Speech-Emotion-Recognition
Speech emotion recognition using both Traditional machine learning models as well Deep learning model using CNN and LSTM and predicting over 7 emotions (Angry, Sad ,Happy , Neutral ,Fear, Disgust and Surprise) .


In this project, I have done my work in a very detailed manner and have explained everything in a simple and step-wise manner. All you need to do is follow the README.md file in a step-by-step manner. We start with downloading and extracting the datasets as they are in a zip file. Next, download the Python file of Traditional Machine Learning models, change the path according to your system, and run all the lines step-by-step. In the next step, download the Python file of Building Various CNN and CNN+LSTM models and do the same – change the data path and run step-by-step. Although I have provided all the results in a tabular manner and have made the necessary comparisons. Next, we have Building our Custom model using CNN and LSTM, in which you can see the model made in the attached figure. I have made two hyperparameter optimizations: one is choosing the Optimizer, and the second is using the concept of callbacks. By doing this, we will achieve our Model named as My Model, in which we will be plotting the Confusion Matrix and Classification Report. We will be checking the accuracy of our model on the validation data, and that is followed by the Conclusion.

PLEASE NOTE: Every time you run a particular model, you will get a slightly different result. It may vary by a small value, so you don't need to bother about why my result is not matching exactly with the result I have provided. Sometimes, you will notice that running the same code again will result in a different accuracy, but the variation will be little, and that would not affect the overall result or conclusion this is due to the dynamic nature of Machine learning and Deep learning Models.

<br>
<br>
<br>

## Dataset
I have used total of 5 datasets these are as:

1.  RAVDESS : Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
 dataset is a comprehensive multimodal resource for emotional expression, con
taining audio, video, and song modalities. For this study, we utilized the audio
only portion, which consists of approx 7356 speech files (16-bit, 48kHz .wav)
 from 24 professional actors (12 female, 12 male).
  https://drive.google.com/file/d/1Yl_-3A3kDpZgnAxfhiXqmcJIQUZDrBne/view?usp=drive_link

<br>

2. TESS:The Toronto Emotional Speech Set (TESS) is a unique dataset developed at
 the University of Toronto, focusing exclusively on female emotional speech. It
 comprises 2,800 utterances from two female actresses, aged 26 and 64 years,
 portraying seven emotional states: anger, disgust, fear, happiness, pleasant
 surprise, sadness, and neutral.
 https://drive.google.com/file/d/1Yk6g-jKWXWRdPMGyaozK5nqOL3WfiAcw/view?usp=drive_link

<br>

3. EmoDB: The Berlin Emotional Speech Database (Emo-DB) is a widely recognized and
 freely available German emotional speech database, created by the Institute of
 Communication Science at the Technical University of Berlin. It contains a
 total of 535 utterances spoken by 10 professional actors (5 male and 5 female),
 ensuring a balanced representation of gender.
  https://drive.google.com/file/d/15fO0LUtZKNIZHqfeXcZ7LdqO3wFNtNbN/view?usp=drive_link

<br>

4. ESD: TheEmotional Speech Dataset (ESD) is a specialized dataset designed for voice
 conversion research, with a particular focus on emotional speech. It consists of
 11
35,000 files, comprising 350 parallel utterances spoken by 20 speakers: 10 native
 English and 10 native Chinese speakers. The dataset covers five emotion categories: neutral, happy, angry, sad, and
 surprise, providing a diverse yet focused set of emotions for analysis. More
 than 29 hours of speech data were meticulously recorded in a controlled acoustic
 environment, ensuring high-quality audio and minimizing external noise factors.
  https://drive.google.com/file/d/1goCK5C2ko_9NzRIk_x_FzKfrBgoi20xi/view?usp=drive_link

<br>

5. CUSTOM:Acustomdataset is To address the under-representation of the emotions disgust
 and fear in existing datasets, a custom dataset was curated, comprising approx
imately 6,000 audio files explicitly labeled for these two emotional states. This
 custom dataset played a crucial role in balancing the distribution of emotions,
 ensuring that the analysis was not biased toward more commonly represented
 emotional states.
 https://drive.google.com/file/d/1gN5M_ZPrQmULjkg5wVMG34xaLeDIw4EI/view?usp=drive_link

I have provided the link to all the dataset in zip files download it to proceed for further steps.

<br>
<br>
<br>

## Comparing Traditional Machine Learning models.

We have made a comparison of various machine learning models for classification
 tasks. The models we have included in our analysis are RandomForestClassifier,
 SVC (Support Vector Classifier), GradientBoostingClassifier, KNeighborsClas
sifier, MLPClassifier (Multi-Layer Perceptron), BaggingClassifier, AdaBoost
Classifier, and DecisionTreeClassifier.
 To evaluate the performance of these models, we have chosen three perfor
mance metrics: Accuracy, F1-score, and Time. These metrics will be calculated
for both the training and testing datasets, with varying training data sizes of
 1%, 10%, and 100% of the available data. This approach will provide insights
 into how the models perform under different data availability scenarios.
 
<br>

#### Here is the link of python file:
[Model Selector ML.ipynb](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Model%20Selector%20ML.ipynb)

<br>
<br>

#### This is the tabular comparsion for all the models when test size=25%
<br>

![ML Models](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/ML%20Models.png)

<br>

Result: Here, upon observing the notebook, you will see that among all the 8 traditional models, the Random Forest Classifier outperforms all the other 7 ML models across all the performance metrics, and it achieved the highest accuracy of 91.07% .

<br>
<br>
<br>

## Building Various CNN and CNN+LSTM models.

We have considered 4 CNN models and 4 CNN+LSTM models with varying the number of CNN and LSTM layers and observed their accuracy.

<br>

#### The description about all the model you can see in the python file.
[CNN & CNN+LSTM.ipynb](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/CNN%20%26%20CNN%2BLSTM.ipynb)

<br>
<br>


#### 1. Architecture of CNN model:
   ![CNN](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/CNN%20Model.png)
<br>
<br>

#### 2. Architecture of CNN+LSTM model:
  ![CNN+LSTM](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/CNN%2BLSTM%20Model.png)

  <br>

The above figures only show the architectures of the models for a particular number of layers; they are only for visualization. However, in the Python file, it has been coded for varying numbers of layers.

This is the tabular comparsion for all the models when test size=25%

<br>
<br>

#### CNN Models Accuracy

![CNN Models](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/CNN.png)

<br>
<br>



#### CNN+LSTM Models Accuracy

![CNN+LSTM Models](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/CNN%2BLSTM.png)

<br>

Result: On observing, we can see that in the case of only CNN models, as the number of layers increases, the accuracy increases. And on combining these CNN models with LSTM, we see a gradual increase in the accuracy. For CNN models alone, we have the highest accuracy of 90.94% when the number of CNN layers is four. And for CNN with LSTM, the highest accuracy achieved is 92.21% when the number of CNN layers is 4, and the number of LSTM layers is 2.

Hence, a common conclusion we can draw is that deep learning models perform better than traditional machine learning models. Also, on increasing the CNN layers, the accuracy increases, and comparing CNN+LSTM with CNN, CNN+LSTM gives higher accuracy. So, in the next step, we will try to build our own model using CNN and LSTM and name it as the Custom Model.

<br>
<br>

## Building our Custom model using CNN and LSTM

#### Description of the Model: 

![Custom Model](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/Custom%20Model.png)

The model combines two CNN branches and an LSTM component for a 7-class classification task. The first CNN branch has two convolutional layers, while the second has six convolutional layers, each with ReLU activation, dropout, and max pooling. The output tensors from both branches are concatenated and passed through two LSTM layers with 128 units each, with L2 regularization. The first LSTM layer returns the full sequence, and the second returns the final output. Finally, the output is flattened, and a dense layer with 7 units and softmax activation is applied to obtain the classification output.

<br>

### Hyperparameter Optimisation:

#### 1. Optimisers:
So after making the model we first have to choose the best optimiser for our project , for this  We have evaluated the model’s performance using a range of optimizers, including Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, and
 Nadam.

#### Here is the python file:
[Optimizers.ipynb](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Optimizers.ipynb)

<br>

Result: On Observing you will find that Optimisers Adamax has the highest accuracy of 96.12%, however we can see in all the optimiser graph none of them are converging although we have achieved a very good accuracy but there may be a scope for improvement in which we can get a converging graph in less computation to save the number of epoch as in previous case all the optimisers have been run over for 100 epochs.so will be using early stopping and learning rate reduction techniques.

<br>
<br>

#### 2. Optimisers with Early stopping and learning rate reduction(also known as callbacks)

We have implemented two types of callbacks: early stopping andlearning
 rate reduction.
 
 • The early stopping callback monitors the validation loss and stops the training
 process if the loss does not improve for a specified number of epochs (in our case,
 10 epochs). This technique helps prevent overfitting and saves computational
 resources by terminating the training process when further improvements are
 unlikely.
 
 • The learning rate reduction callback adjusts the learning rate dynamically dur
ing training. If the validation loss does not decrease for a certain number of
 epochs (in our case, 5 epochs), the learning rate is reduced by a specified factor
 (0.5 in our implementation). This approach allows the model to escape local
 minima and potentially converge to a better solution.

 #### Here is the python file:
 
[Optimizers(EC&LR).ipynb](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Optimizers(EC%26LR).ipynb)

<br>


#### Without Callback: Accuracy=96.12%

![Without Callback](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/WOC.png)

<br>
<br>

#### With Callbacks: Accuracy=96.48%

![With Callback](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/WC.png)

<br>

Result: On obeserving we can see that Optimiser Adamax with Early stopping and learning rate reduction has given a accuracy of 96.48% in 100 epoch , so we have achieved greater accuracy as well as our model is also conserved.

So we will saving our model with Adamax optimisers with Callbacks.

<br>
<br>

### My Model

So now we have decided our model will be our custom model with the Adamax optimizer and with callback techniques. Moreover, we will run our model for 100 epochs.

#### Here is the Python file :

[My Model.ipynb](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/My%20model.ipynb)

<br>

#### Final Accuracy of Our Model = 96.25%(but in 77 epochs only so saves our computation time)

As told above there may be slight variations each time you run but will not affect the over all conclusion this time our model converged only in 77 epoch you can check in the python file this is due to the dynamic nature of deep learning techniques

So now we will be plotting some Matrix and Reports

<br>
<br>

### Confusion Matrix

It shows the number of prediction made correctly over various emotions , it can be represented in two forms With Normalisation and Without Normalisation . Without Normalisation it shows the number of emotions being predicted correctly but in With Normalisation this number shown in percentage(%).

#### Without Normalisation
![Without Normalisation](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/confusion_matrix(WON).png)

<br>
<br>

#### With Normalisation
![With Normalisation](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/confusion_matrix(WN).png)

<br>
<br>

### CLassification Report

![CLassification Report](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/Classification%20Report.png)

<br>
<br>

### Dataset wise Accuracy

I have not provided the code here because this is very simple. All you need to do is change the data paths in the python file of My model.ipynb. Each time, give the path to one dataset and observe the accuracy, although I have provided the results

![Dataset wise Accuracy](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Images/Dataset%20wise.png)

<br>
<br>

### Validation Accuracy:

Validation accuracy is needed to show that how our model is performing on Unseen data and it also checks the problem of overfitting.

#### Here is the pyhton file:
[Validation.ipynb](https://github.com/Nemesis9450/Speech-Emotion-Recognition/blob/main/Validation.ipynb)

Result: Validation Accuracy is 95.41%

<br>
<br>
<br>

## Conclusion:
 This study undertook a comprehensive evaluation of both traditional machine learn
ing models and deep learning techniques for the task of speech emotion recognition.
 Eight widely-used traditional machine learning algorithms, including Random For
est, Support Vector Machines, Gradient Boosting, K-Nearest Neighbors, Multi-Layer
 Perceptrons, Bagging, AdaBoost, and Decision Trees, were assessed and compared on
 performance metrics such as accuracy, F1-score, and computational time.
 The results showed that the Random Forest classifier achieved the best perfor
mance, with a testing accuracy of 91.07%, outperforming the other traditional ma
chine learning models.

 To further enhance the speech emotion recognition capabilities, a custom deep
 learning architecture was developed, leveraging the synergistic combination of Con
volutional Neural Networks (CNNs) and Long Short-Term Memory (LSTMs). This
 hybrid model, which concatenated the features from two parallel CNN branches before
 feeding them into the LSTM layers, demonstrated remarkable results. The custom
 deep learning model achieved a testing accuracy of 96.48% and a validation accuracy
 of 95.74%, significantly outperforming not only the traditional machine learning ap
proaches but also the sequential CNN and CNN-LSTM models, which had accuracies
 of 90.94% and 92.21%, respectively.
 
 The deep learning-based approach demonstrated clear advantages over the tradi
tional machine learning models, showcasing its ability to effectively capture the low
 level feature extraction , complex patterns in speech data. These results highlight
 the value of exploring advanced modeling techniques, like the synergistic combination
 of CNNs and LSTMs, in this project our custom model has demonstrated superior
 performance over both traditional machine learning models and other deep learning
 architectures, including the sequential CNN and the sequential CNN-LSTM combi
nation



   
   












