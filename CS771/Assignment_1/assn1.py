import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def slip_data_for_different_models( data ):
    for row in data:
        a = row[64]*8+row[65]*4+row[66]*2+row[67]
        b = row[68]*8+row[69]*4+row[70]*2+row[71]
        if a>b:
            for i in range(4):
                row[68+i],row[64+i]=row[64+i],row[68+i]
            row[72] = 1 - row[72]
    
    arr = []

    for i in range(256):
        arr.append([np.zeros(65)])
    for row in data:
        a = row[64]*8+row[65]*4+row[66]*2+row[67]
        b = row[68]*8+row[69]*4+row[70]*2+row[71]
        a = int(16*a+b)
        arr[a].append(np.append(row[:64], row[72]))
    
    return arr


################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
    
    new_data = slip_data_for_different_models(Z_train)
    models = []
    for d2 in new_data:
        d = d2[1:]
        if (len(d)):
            x = np.array(d)
            clf = LinearSVC( loss = "squared_hinge", max_iter=1e7, penalty='l2', dual=False, tol=1e-4 )
            clf.fit( x[:,:-1], x[:,-1] )
            models.append(clf)
        else:
            models.append(None)
    return models

################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	
    i = 0
    pred = np.zeros(X_tst.shape[0])
    for row in X_tst:
        a = row[64]*8+row[65]*4+row[66]*2+row[67]
        b = row[68]*8+row[69]*4+row[70]*2+row[71]
        f = 1
        if a>b:
            a,b=b,a
            f = -1
        x = model[int(16*a+b)].predict([row[:64]])
        if f == -1:
            x[0] = 1- x[0]
        pred[i] = x[0]
        i = i + 1
    return pred
