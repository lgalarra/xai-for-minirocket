#!/usr/bin/env python
# coding: utf-8

# In[41]:
import os
# Must be set before importing joblib/sklearn
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")   # force threading backend
os.environ.setdefault("JOBLIB_START_METHOD", "threading")

from sklearn.ensemble import RandomForestClassifier



import importlib

import minirocket_multivariate_variable as mmv

importlib.reload(mmv)
from sklearn.linear_model import LogisticRegression
from utils import get_cognitive_circles_data, get_cognitive_circles_data_for_classification, prepare_cognitive_circles_data_for_minirocket
from classifier import MinirocketClassifier


# ## Fetching data

# In[42]:


(X_train, y_train), (X_test, y_test) = get_cognitive_circles_data_for_classification('../data/cognitive-circles',
                                                                                     target_col='RealDifficulty',
                                                                                     as_numpy=True)


# ## Training MiniROCKET

# In[71]:


#LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
classifier = MinirocketClassifier(minirocket_features_classifier=RandomForestClassifier())
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
print('Accuracy score:', accuracy_score(y_test, classifier.predict(X_test)))


# ## Explain one instance

# In[72]:


IDX_TO_EXPLAIN = 0


# In[73]:



explainer =  classifier.get_explainer(X=X_train, y=classifier.predict(X_train))


# In[76]:

explanation = explainer.explain_instances(X_test[0], classifier_explainer='extreme_feature_coalitions',
                                          reference_policy='opposite_class_medoid')


# In[66]:
for i in range(2):
    print('Logits:', explanation.instances[i]['instance_logits'], explanation.instances[i]['reference_logits'])
    print('Predictions:', explanation.instances[i]['instance_prediction'], explanation.instances[i]['reference_prediction'])
    print(explanation.instances[i]['coefficients'].sum())
    print(explanation.instances[i]['minirocket_coefficients'].sum())

print(explanation.check_explanations_local_accuracy(tol=1e-4).all())
print(explanation.check_explanations_local_accuracy(tol=1e-4).any())


explanations_p2p = classifier.explain_instances(X_test[0], explanation.get_references(), explainer='extreme_feature_coalitions')
# In[21]:



