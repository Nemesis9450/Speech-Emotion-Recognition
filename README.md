# Speech-Emotion-Recognition
Speech emotion recognition using both Traditional machine learning models as well Deep learning model using CNN and LSTM and predicting over 7 emotions (Angry, Sad ,Happy , Neutral ,Fear, Disgust and Surprise) .


In this project i have made a model to recognise the emotion from speech, i have made a 
## Dataset
I have used total of 5 datasets these are as:

1.  RAVDESS : Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
 dataset is a comprehensive multimodal resource for emotional expression, con
taining audio, video, and song modalities. For this study, we utilized the audio
only portion, which consists of approx 7356 speech files (16-bit, 48kHz .wav)
 from 24 professional actors (12 female, 12 male). 

2. TESS:The Toronto Emotional Speech Set (TESS) is a unique dataset developed at
 the University of Toronto, focusing exclusively on female emotional speech. It
 comprises 2,800 utterances from two female actresses, aged 26 and 64 years,
 portraying seven emotional states: anger, disgust, fear, happiness, pleasant
 surprise, sadness, and neutral.

3. EmoDB: The Berlin Emotional Speech Database (Emo-DB) is a widely recognized and
 freely available German emotional speech database, created by the Institute of
 Communication Science at the Technical University of Berlin. It contains a
 total of 535 utterances spoken by 10 professional actors (5 male and 5 female),
 ensuring a balanced representation of gender.

4. ESD: TheEmotional Speech Dataset (ESD) is a specialized dataset designed for voice
 conversion research, with a particular focus on emotional speech. It consists of
 11
35,000 files, comprising 350 parallel utterances spoken by 20 speakers: 10 native
 English and 10 native Chinese speakers. The dataset covers five emotion categories: neutral, happy, angry, sad, and
 surprise, providing a diverse yet focused set of emotions for analysis. More
 than 29 hours of speech data were meticulously recorded in a controlled acoustic
 environment, ensuring high-quality audio and minimizing external noise factors.

5. CUSTOM:Acustomdataset is To address the under-representation of the emotions disgust
 and fear in existing datasets, a custom dataset was curated, comprising approx
imately 6,000 audio files explicitly labeled for these two emotional states. This
 custom dataset played a crucial role in balancing the distribution of emotions,
 ensuring that the analysis was not biased toward more commonly represented
 emotional states.
 Link: https://drive.google.com/file/d/1gN5M_ZPrQmULjkg5wVMG34xaLeDIw4EI/view?usp=drive_link

I have provided the link to all the dataset in zip files download it to proceed for further steps.


## Comparing Traditional Machine Learning models.

Wehavemadeacomparison of various machine learning models for classification
 tasks. The models we have included in our analysis are RandomForestClassifier,
 SVC (Support Vector Classifier), GradientBoostingClassifier, KNeighborsClas
sifier, MLPClassifier (Multi-Layer Perceptron), BaggingClassifier, AdaBoost
Classifier, and DecisionTreeClassifier.
 To evaluate the performance of these models, we have chosen three perfor
mance metrics: Accuracy, F1-score, and Time. These metrics will be calculated
 14
for both the training and testing datasets, with varying training data sizes of
 1%, 10%, and 100% of the available data. This approach will provide insights
 into how the models perform under different data availability scenarios.

Here is the link of python file:


This is the tabular comparsion for all the models when test size=25%

Result: Here, upon observing the notebook, you will see that among all the 8 traditional models, the Random Forest Classifier outperforms all the other 7 ML models across all the performance metrics, and it achieved the highest accuracy of 91.07% .

## Building Various CNN and CNN+LSTM models.

We have considered 4 CNN models and 4 CNN+LSTM models with varying the number of CNN and LSTM layers and observed their accuracy.
1. Architecture of CNN model:
   
2. Architecture of CNN+LSTM model:


The above figures only show the architectures of the models for a particular number of layers; they are only for visualization. However, in the Python file, it has been coded for varying numbers of layers.

This is the tabular comparsion for all the models when test size=25%

The description about all the model you can see in the python file.

## Building our Custom model using CNN and LSTM

Description of the Model: 


The model combines two CNN branches and an LSTM component for a 7-class classification task. The first CNN branch has two convolutional layers, while the second has six convolutional layers, each with ReLU activation, dropout, and max pooling. The output tensors from both branches are concatenated and passed through two LSTM layers with 128 units each, with L2 regularization. The first LSTM layer returns the full sequence, and the second returns the final output. Finally, the output is flattened, and a dense layer with 7 units and softmax activation is applied to obtain the classification output.









