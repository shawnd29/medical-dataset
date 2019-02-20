'''
This project is focused on evaluating the 15 casual features from a group of 30,000 datasets of single Nucleotide Polymorphism (SNP) genotypes.
This was done by estimating the goodness of fit using the Chi-square and then predicting the model.

To run this program type python medical.py <dataset> <trainlabels>

Shawn D'Souza
'''



import sys
import csv
import random as rnd
from sklearn import svm
from sklearn.model_selection import train_test_split

csv.register_dialect('pipes', delimiter=' ', escapechar='\n')
##-----Load the data and class labels
# read file and return a 2d matrix
def read_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, dialect='pipes')
        data = []
        for rows in reader:
            temp = []
            for col in rows:
                if col != '':
                    temp.append(int(col))
            data.append(temp)
        return data

## read the label file and return a vector
def extract_trainlabels(filename):
    labl = read_file(filename)
    L = [labl[i][0] for i in range(0, len(labl))]
    return L

##-----Chi-square feature selection
def chi2(X, y, top):
    rows = len(X)
    cols = len(X[0])
    T = []
    for j in range(0, cols):
        ct = [[1,1],[1,1],[1,1]]    
        for i in range(0, rows):
            if y[i] == 0:
                if X[i][j] == 0:
                    ct[0][0] += 1
                elif X[i][j] == 1:
                    ct[1][0] += 1
                elif X[i][j] == 2:
                    ct[2][0] += 1
            elif y[i] == 1:
                if X[i][j] == 0:
                    ct[0][1] += 1
                elif X[i][j] == 1:
                    ct[1][1] += 1
                elif X[i][j] == 2:
                    ct[2][1] += 1
        col_totals = [ sum(x) for x in ct]
        row_totals = [ sum(x) for x in zip(*ct) ]
        total = sum(col_totals)
        exp_value = [[(row*col)/total for row in row_totals] for col in col_totals]
        sqr_value = [[((ct[i][j] - exp_value[i][j])**2)/exp_value[i][j] for j in range(0,len(exp_value[0]))] for i in range(0,len(exp_value))]
        x_2 = sum([sum(x) for x in zip(*sqr_value)])
        T.append(x_2)
    indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)
    idx = indices[:top]
    return idx

##-----Extract top 15 features
# X: dataset
# cols: columns number with highest chi-square score
# returns: new dataset with 15 features
def feature_extraction(X, cols):
    V = []
    columns = list(zip(*X))
    for j in cols:
        V.append(columns[j])
    V = list(zip(*V))
    return V



train_data = sys.argv[1]        # training dataset
train_labels = sys.argv[2]      # training labels
test_data = sys.argv[3]         # testing dataset

##-----build data for svm.svc
X_train = read_file(train_data)                 #read training data 
y_train = extract_trainlabels(train_labels) #read training data labels

idx = chi2(X_train, y_train, 15)                #perform chi-square and get index of top 15 features

X_train = feature_extraction(X_train, idx)      #create new training dataset using top 15 features

X_test = read_file(test_data)                   #read training data file
X_test = feature_extraction(X_test, idx)        # create new testing dataset using top 15 features

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=1.0, gamma=1) 
model.fit(X_train, y_train)
model.score(X_train, y_train)
#Predict Output
predicted = model.predict(X_test)

# fw = open("predicted_labels.txt", "w")
for x in range(0,len(predicted),1):
    # fw.write(str(predicted[x])+" "+str(x)+"\n")       
    print (predicted[x],x)  

print('Total number of features used: 15')
print('Feature column numbers used in prediction: {}'.format(idx))