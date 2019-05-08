
# coding: utf-8

# # Gender Classification
# Ibrahim Mohammad (1618923)
# imohammad@uh.edu
#

# ### Import Necessary Sklearn and pandas packages

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC


# ### Imports for Visualization

# In[2]:


from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.features import Manifold
from yellowbrick.text import TSNEVisualizer
from yellowbrick.model_selection import LearningCurve


# ## Reading the data
# There are exactly 6 documents which are null, we need to drop those records. Target labels are mix of both upper and lower case characters and few target classes are padded with spaces. We need to trim and convert to either upper case or lower case characters. Otherwise there is a huge difference in prediction.

# In[3]:


df = pd.read_excel('blog-gender-dataset.xlsx', sheet_name='training', header = None, usecols=[0,1],  names=['blog', 'gender'])
tdf = df.dropna().copy();
tdf['blog'] = tdf['blog'].astype(str)
tdf["gender"] = tdf["gender"].map(lambda x: 1 if x.strip().lower()=='m' else 0)


# ## Preprocessing

# In[4]:


def convertSpecial(row):
    row = row.replace('???', ' foundQQQMark ')
    row = row.replace('??', ' foundQQMark ')
    row = row.replace(':)', ' foundHsmiley ')
    row = row.replace(':(', ' foundSsmiley ')
    row = row.replace('!!!', ' foundEEEmark ')
    row = re.sub(r'(\d+\.\d+)|(\d+x\d)|(\d+)|(\_)', '', row)
    return row

tdf['blog'] = tdf['blog'].map(convertSpecial);


# ## Vectorizing
# Vectorize articles based on bigram and remove stop words. Considering top 100,000 features.

# In[5]:


tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    smooth_idf=True,
    max_features=100000
)
vec = tfidf.fit_transform(tdf['blog'])
print('Initial Vector Size', vec.shape)


# ## Feature Selection from Model

# In[8]:


from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import StratifiedKFold
sfmodel_1 = SelectFromModel(SGDClassifier(loss='hinge', penalty='elasticnet',alpha=0.0001, random_state=42,max_iter=500, tol=1e-3).fit(vec, tdf['gender']), prefit=True)
newVec = sfmodel_1.transform(vec)
print("Intermediate Vector size", newVec.shape)
sfmodel_2 = SelectFromModel(SGDClassifier(loss='hinge', penalty='elasticnet',alpha=0.0001, random_state=42,max_iter=500, tol=1e-3).fit(newVec, tdf['gender']), prefit=True)
finalVec = sfmodel_2.transform(newVec)
print("Final Vector size", finalVec.shape)


# ## Splitting the data in 7:3 ratio as Train and Test. Stratified based on gender

# In[9]:


X_train ,X_test, y_train, y_test = train_test_split(finalVec, tdf['gender'], test_size=0.30, stratify=tdf['gender'])


# ## Grid Search Cross Validation
# Creating a pipeline for Hyperparameter tuning using GridSearch cross validation

# In[ ]:


steps = [('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42,max_iter=500, tol=1e-3))]
text_clf = Pipeline(steps)
parameters = {
    'clf__alpha': (0.0001,0.001, 0.01),
    'clf__penalty':('l2', 'elasticnet')
}
gs_clf = GridSearchCV(text_clf, parameters, cv=10, iid=False, n_jobs=-1)


# ## Training the model and Predicting target class

# In[24]:


gs_clf.fit(X_train.toarray(), y_train)
y_pred = gs_clf.predict(X_test.toarray())
print("Classification Report:", metrics.classification_report(y_test, y_pred))
print("Accuracy Score: ", metrics.accuracy_score(y_test, y_pred))
print("Number of Train features: ", X_train.shape)
print("Number of Test features: ", X_test.shape)


# In[11]:


print("Best Score", gs_clf.best_score_)
print("Best Parameters", gs_clf.best_params_)
print("Stats for the model", gs_clf.cv_results_)


# ## Classification Report

# In[12]:


classes = ["Female", "Male"]
cr_viz = ClassificationReport(gs_clf, classes=classes, support=True)
cr_viz.fit(X_train.toarray(), y_train)  # Fit the visualizer and the model
cr_viz.score(X_test.toarray(), y_test)  # Evaluate the model on the test data
cr_viz.poof(outpath="graphs/cr.png")
cr_viz.poof()


# ## Confusion Matrix

# In[13]:


cm_viz = ConfusionMatrix(gs_clf)
cm_viz.fit(X_train.toarray(), y_train)
cm_viz.score(X_test.toarray(), y_test)
cm_viz.poof(outpath="graphs/cm.png")
cm_viz.poof()


# ## Area Under the Curve

# In[14]:


auc_viz = ROCAUC(gs_clf, classes=['Female', 'Male'], micro=False, macro=False, per_class=False)
auc_viz.fit(X_train, y_train)  # Fit the training data to the visualizer
auc_viz.score(X_test, y_test)  # Evaluate the model on the test data
auc_viz.poof(outpath="graphs/auroc.png")
auc_viz.poof()


# ## Precision Recall Curve

# In[15]:


prc_viz = PrecisionRecallCurve(gs_clf, iso_f1_curves=True,per_class=True,fill_area=True, micro=False)
prc_viz.fit(X_train, y_train)
prc_viz.score(X_test, y_test)
prc_viz.poof(outpath="graphs/prc.png")
prc_viz.poof()


# ## Class Prediction Error

# In[16]:


cpe_viz = ClassPredictionError(gs_clf, classes=classes)
cpe_viz.fit(X_train, y_train)
# Evaluate the model on the test data
cpe_viz.score(X_test, y_test)
# Draw visualization
cpe_viz.poof(outpath="graphs/cpe.png")
cpe_viz.poof()


# In[17]:


sizes = np.linspace(0.3, 1.0, 10)
lr_viz = LearningCurve(gs_clf,cv=10, train_sizes=sizes,
    scoring='f1_weighted', n_jobs=4 )
lr_viz.fit(finalVec, tdf['gender'])
lr_viz.poof(outpath="graphs/lr.png")
lr_viz.poof()


# ### TSNE projection of total documents
# Projection of document similarity onto 2 dimensional space

# In[18]:


tsne = TSNEVisualizer()
tsne.fit(finalVec, tdf['gender'])
tsne.poof(outpath="graphs/tsnedoc.png")
tsne.poof()


# ## Implementation using Deep Learning
# Import Necessary Keras packages to create a Neural Network

# In[25]:


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# ### Create a Sequential Model with 6 hidden layers each with 100 nodes
# Each hidden layer uses ReLU activation function and Output layer uses sofmax activation. Stopping criteria for the model is after no change in 2 successive epochs.

# In[20]:


model = Sequential()
#Convert target labels to categorical data for classification
target = to_categorical(tdf['gender'])
#Building a deep neural network of 6 hidden layers each with 100 units and each with ReLU activation function
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
#Create output layer of 2 nodes with softmax activation function
model.add(Dense(units=2, activation='softmax'))


# In[21]:


#Stopping criteria for the model after no change in 2 successive epochs
early_stopping_monitor = EarlyStopping(patience=2)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(finalVec.toarray(), target, epochs=10, validation_split=0.3, callbacks=[early_stopping_monitor])

