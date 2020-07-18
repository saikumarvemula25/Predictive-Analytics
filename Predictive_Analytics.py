# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning) 
df = pd.read_csv("data.csv")
X = df.iloc[:,0:df.shape[1]-1]
y = df.iloc[:,df.shape[1]-1:df.shape[1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, stratify=y)

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    b= y_true[y_true.columns[len(y_true.columns)-1]].nunique()
    ground=y_true.values.ravel()
    pred=y_pred
    pred = pred.reshape(pred.shape[0],1)
    ground = ground.reshape(ground.shape[0],1)
    a=np.multiply(ground,b)
    k=np.add(a,pred)
    hist=np.histogram(k,bins=b**2)
    w = hist[0].reshape(b,b)
    x = np.diag(w)
    a=np.sum(x)
    b=np.sum(w,axis=1)
    b1=np.sum(b) 
    acc = a/b1
    return acc

def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
        b= y_true[y_true.columns[len(y_true.columns)-1]].nunique()
        ground=y_true.values.ravel()
        pred=y_pred
        pred = pred.reshape(pred.shape[0],1)
        ground = ground.reshape(ground.shape[0],1)
        a=np.multiply(ground,b)
        k=np.add(a,pred)
        hist=np.histogram(k,bins=b**2)
        w = hist[0].reshape(b,b)
        x = np.diag(w)
        u=np.sum(w,axis=1)
        recall=np.divide(x,u)
        return recall

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    b= y_true[y_true.columns[len(y_true.columns)-1]].nunique()
    ground=y_true.values.ravel()
    pred=y_pred
    pred = pred.reshape(pred.shape[0],1)
    ground = ground.reshape(ground.shape[0],1)
    a=np.multiply(ground,b)
    k=np.add(a,pred)
    hist=np.histogram(k,bins=b**2)
    w = hist[0].reshape(b,b)
    x = np.diag(w)
    u=np.sum(w,axis=0)
    precision=np.divide(x,u)
    return precision


def WCSS(Clusters,newcentroids):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    wcss = []
    k = newcentroids.values.shape[0]
    for t in range(1, k+1):
        t1 = Clusters[Clusters.cluster.eq(t)]
        t1 = t1.drop('cluster',1)
        t2 = newcentroids.iloc[t-1]
        inctiledata = np.tile(t2,(t1.shape[0],1))
        t3 = np.sum((t1-inctiledata)**2)
        t4 = np.mean(t3)
        wcss.append(t4)
    wcss = sum(wcss)
    return wcss    
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    b= y_true[y_true.columns[len(y_true.columns)-1]].nunique()
    ground=y_true.values.ravel()
    pred=y_pred
    pred = pred.reshape(pred.shape[0],1)
    ground = ground.reshape(ground.shape[0],1)
    a=np.multiply(ground,b)
    k=np.add(a,pred)
    hist=np.histogram(k,bins=b**2)
    cmatrix = hist[0].reshape(b,b)
    return cmatrix

def KNN(X_train,X_test,Y_train,k):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
        tx1 = X_train.copy()
        tx2 = Y_train.copy()
        tx1['actualvalue'] = tx2 
        finalans = []
        for index,row in X_test.iterrows():
            row = row.values.reshape(1,row.shape[0])
            t = np.tile(row,(X_train.shape[0],1))
            x1 = np.sqrt(np.sum((X_train - t)**2,1))
            x1 = x1.to_frame()
            tx1['Edistance'] = x1
            adddist1 = tx1.sort_values('Edistance')
            ndata = adddist1.iloc[0:k]
            ndata2 = ndata.groupby('actualvalue')['actualvalue'].count()
            ndata2 = ndata2.sort_values(ascending= False)
            test =  ndata2.keys()
            ans = test[0]
            finalans.append(ans)
        return finalans
    

def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    finalarr = []
    forest = []
    ntree = 5
    splitdata = X_train.copy()
    splitdatay = Y_train.copy()
    splitdata = pd.concat([splitdata,splitdatay],1)
    for i in range(ntree):
        xsplitd = splitdata.sample(frac = 1, replace=True,random_state=1)  #random sampling for decision tree
        count = 0
        features = np.sqrt(X_train.shape[1])
        features = math.floor(features)
        depth =7
        counter = 0
        tree = buildtree(features,xsplitd,counter,depth)
        forest.append(tree)
    
    for dataxv in X_test.index:
        example = df.iloc[dataxv:dataxv+1,0:df.shape[1]]
        t = []
        l = []
        for ids in forest:
            tree = ids
            x = predict_example(example, tree,l)
            t.append(x)
        b1 = max(set(l), key=l.count)
        finalarr.append(b1)
    return finalarr



def buildtree(features,data,counter,depth):
    x = data[data.columns[len(data.columns)-1]].value_counts()
    #print(type(x))
    if counter == depth:
        klk = 0
    else:
        counter = counter +1
        sortedarray = bestsplit(features,data)
        lfilter,rfilter = filterdata(sortedarray,data)
        dummy = data.copy()
        datarec = dummy.groupby(dummy.columns[len(dummy.columns)-1]).size().reset_index(name='count')
        datahan = {}
        for i,n in datarec.iterrows():
            datahan1 ={}
            z = list(n.to_dict().values())
            datahan1[z[0]] = z[1]
            datahan.update(datahan1)
        featureandcon = "{}!<=!{}!{}".format(sortedarray[0], data[sortedarray[0]].mean(),datahan)   #added 
        stree = {featureandcon: []}
        ltree = ""
        rtree = ""
        if lfilter.shape[0]>0:
            ltree = buildtree(features,lfilter,counter,depth)
        if rfilter.shape[0]>0:
            rtree = buildtree(features,rfilter,counter,depth)
        stree[featureandcon].append(ltree)
        stree[featureandcon].append(rtree)

        return stree           #added

def filterdata(sortedarrayfinal,data):
    lfilter = data[data[sortedarrayfinal[0]]<=data[sortedarrayfinal[0]].mean()].copy()
    rfilter = data[data[sortedarrayfinal[0]]>data[sortedarrayfinal[0]].mean()].copy()
    return lfilter,rfilter
   
def bestsplit(features,data):
        x3 = data.copy()
        x4 = x3.iloc[:,0:data.shape[1]-2]
        x1 = x4.sample(features,axis = 1)
        x2 = x1.mean(axis = 0)
        dict1 = x2.keys()
        dict2 = x2.values
        dictsort = {}
        sorteddictsort = {}
        for k,v in zip(dict1,dict2):
            columnvalue = data[k]
            columnvalue = columnvalue.to_frame()
            targetvalue = Y_train.copy()
            lfilter = data[data[k]<=v]
            leftgini = giniindex(lfilter)
            rfilter = data[data[k]>v]
            rightgini = giniindex(rfilter)
            if lfilter.shape[0]+rfilter.shape[0] == 0:
                weightedavg = 0
            else:
                weightedavg = ((lfilter.shape[0]/(lfilter.shape[0]+rfilter.shape[0]))*leftgini)+((rfilter.shape[0]/(lfilter.shape[0]+rfilter.shape[0]))*rightgini)
            dictsort[k]=weightedavg
        sorteddictsort = sorted(dictsort.items(), key = operator.itemgetter(1))
        t = list(sorteddictsort[0])
        return t
    
def giniindex(lfilter):
    totalrec = lfilter.shape[0]
    gpy = lfilter[lfilter.columns[len(lfilter.columns)-1]].nunique()
    t = lfilter.columns[len(lfilter.columns)-1]
    d = lfilter[t].value_counts()
    d = np.subtract(1,np.sum((d/totalrec)**2))
    d = d.astype('float')
    return d

def predict_example(example, tree, l):
    if tree!=None:
        treelist = list(tree.values())
        treelist1 = treelist[0]
        x34 = str(treelist1[0])
        x45 = str(treelist1[1])
        fv = []
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value, data = question.split("!",4)
        data = eval(data)
        feature_name = int(feature_name)

        if x34 =="None" and x45 == "None":
            fv = max(data.items(), key = operator.itemgetter(1)) 
            ret = fv[0]
            l.append(fv[0])
    
            
        if comparison_operator == "<=":
            if example.iloc[0][feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        residual_tree = answer
        predict_example(example, residual_tree,l)

    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    c=np.cov(np.transpose(X_train))    
    w,v=np.linalg.eig(c)
    wi=np.argsort(w)
    wi_d=wi[::-1]
    k=wi_d[0:N]
    s = X_train.shape[1]
    xj = np.zeros((s,1))
    V=np.mat(xj)
    for j in k:
        col=np.array(v[:,[j]])
        V=np.column_stack((V,col))
    V=V[:,1:]
    P=np.dot(X_train,V)
    return P
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    rows = X_train.shape[0]-1
    rows = int(rows)
    rand = np.random.randint(0,rows)
    #knndata = X_train.iloc[:,0:df.shape[1]-2]
    centroids = X_train.iloc[[rand]]
    getoutput = []
    getoutput1 = [0]
    appendcentroid1 = X_train.copy()
    appendcentroid = X_train.copy()
    finallist = X_train.copy()
    for i in range(N-1):                                         #initialize centroids
        randrow = np.random.randint(0,rows)
        df2 = X_train.iloc[[randrow]]
        centroids = centroids.append(df2, ignore_index = True)
        isoptimized = True  # loopcondition initialize

    while isoptimized:
    
        Ed = centroids.iloc[[0]]
        Ed = np.tile(Ed,(df.shape[0],1))
        distances = np.sqrt(np.sum((appendcentroid1-Ed)**2,1))
        distances = distances.to_frame()
        for c in range(1,N):
            Ed = centroids.iloc[[c]]
            Ed = np.tile(Ed,(df.shape[0],1))
            d1 = np.sqrt(np.sum((appendcentroid-Ed)**2,1))
            d1 = d1.to_frame()
            distances = pd.concat([distances,d1],1,ignore_index = True)

        new_labels = distances.idxmin(axis=1)
        new_labels = new_labels+1
        appendcentroid1['cluster'] = new_labels
        newcentroids = appendcentroid1.groupby('cluster').mean()
        for t in range(1, N+1):
            t1 = appendcentroid1[appendcentroid1.cluster.eq(t)].copy()
            t1 = t1.drop('cluster',1)
            t2 = newcentroids.iloc[t-1]
            inctiledata = np.tile(t2,(t1.shape[0],1))
            t3 = np.sum((t1-inctiledata)**2)
            t4 = np.mean(t3)
            getoutput.append(t4)
        getoutput1.append(sum(getoutput))
        x1 = getoutput1[-1]
        x2 = getoutput[-2]
        finallist = appendcentroid1.copy()
        appendcentroid1 = appendcentroid1.drop('cluster', 1)
        print(finallist)

        if (x1-x2)<1000000:
            centroids = newcentroids
        else:
            isoptimized = False
    return finallist
        
        
def SklearnSupervisedLearning(X_train,Y_train,X_test,Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    logisticmodel = LogisticRegression(solver = 'lbfgs',max_iter = 10000)
    logisticmodel.fit(X_train,Y_train.values.ravel())
    predict = logisticmodel.predict(X_test)
    logisticaccuracy = metrics.accuracy_score(predict,Y_test)
    #print(model.score(testdata,testtarget)
    
    svmmodel = SVC(kernel = 'linear')
    svmmodel.fit(X_train,Y_train.values.ravel())
    svmpredict = svmmodel.predict(X_test)
    svmaccuracy = metrics.accuracy_score(svmpredict,Y_test)
    
    knnmodel = KNeighborsClassifier(n_neighbors = 100, weights = 'distance')
    knnmodel.fit(X_train,Y_train.values.ravel())
    knnpredict = knnmodel.predict(X_test)
    knnaccuracy = metrics.accuracy_score(Y_test,knnpredict)
    
    Dtreemodel = DecisionTreeRegressor(max_depth = 8)
    Dtreemodel.fit(X_train,Y_train.values.ravel())
    Dtreepredict = Dtreemodel.predict(X_test)
    #print(Dtreemodel.score(testdata,testtarget))
    Dtreeaccuracy = metrics.accuracy_score(Y_test,Dtreepredict)
    
    return logisticaccuracy,svmaccuracy,knnaccuracy,Dtreeaccuracy


def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    estimators = []
    model1 = LogisticRegression(max_iter=35000,multi_class='auto',solver='lbfgs')
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC(kernel = 'linear')
    estimators.append(('svm', model3))
    model4 = KNeighborsClassifier(n_neighbors=8)
    estimators.append(('KNN', model4))
    votes = VotingClassifier(estimators)
    votes=votes.fit(X_train,Y_train.values.ravel())
    predictionvotes=votes.predict(testdata)
    accuracy = metrics.accuracy_score(testtarget,predictionvotes)
    return accuracy

def matplot(X_train,Y_train,X_test,Y_test):
    fig=plt.figure(figsize=(20,6))
    logisticmodel = LogisticRegression(solver = 'lbfgs',max_iter = 10000)
    logisticmodel.fit(X_train,Y_train.values.ravel())
    predict = logisticmodel.predict(X_test)
    confusionlogistic=ConfusionMatrix(Y_test,predict)
    plt.subplot(241) 
    plt.imshow(confusionlogistic)
    plt.title('Logistic regression confusion matrix')
    plt.xlabel('y predicted from Logistic regression')
    plt.ylabel('Actual y')
    
    svmmodel = SVC(kernel = 'linear')
    svmmodel.fit(X_train,Y_train.values.ravel())
    svmpredict = svmmodel.predict(X_test)
    confusionsvm=ConfusionMatrix(Y_test,svmpredict)
    plt.subplot(242) 
    plt.imshow(confusionsvm)
    plt.title('SVM confusion matrix')
    plt.xlabel('y predicted from SVM')
    plt.ylabel('Actual y')

    knnmodel = KNeighborsClassifier(n_neighbors = 100, weights = 'distance')
    knnmodel.fit(X_train,Y_train.values.ravel())
    knnpredict = knnmodel.predict(X_test)
    confusionknn=ConfusionMatrix(Y_test,knnpredict)
    plt.subplot(243) 
    plt.imshow(confusionknn)
    plt.title('knn confusion matrix')
    plt.xlabel('y predicted from knn')
    plt.ylabel('Actual y')

    
    Dtreemodel = DecisionTreeRegressor(max_depth = 8)
    Dtreemodel.fit(X_train,Y_train.values.ravel())
    Dtreepredict = Dtreemodel.predict(X_test)
    confusionDtree=ConfusionMatrix(Y_test,Dtreepredict)
    plt.subplot(244) 
    plt.imshow(confusionDtree)
    plt.title('Decisiontree confusion matrix')
    plt.xlabel('y predicted from Decisiontree')
    plt.ylabel('Actual y')
    
    estimators = []
    model1 = LogisticRegression(max_iter=35000,multi_class='auto',solver='lbfgs')
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC(kernel = 'linear')
    estimators.append(('svm', model3))
    votes = VotingClassifier(estimators)
    votes=votes.fit(data,target.values.ravel())
    predictionvotes=votes.predict(testdata)
    confusionvotes=ConfusionMatrix(Y_test,predictionvotes)
    plt.subplot(245) 
    plt.imshow(confusionvotes)
    plt.title('Voting Classifier confusion matrix')
    plt.xlabel('y predicted from voting classifier')
    plt.ylabel('Actual y')
    

def hypergridparameter(X_train,Y_train,X_test,Y_test):
    model = SVC(kernel = 'linear') 
    model.fit(X_train, Y_train.values.ravel()) 
  
    predictions = model.predict(X_test) 
    parameters = {'C': [1, 10, 100],  
                  'gamma': [0.1, 0.01, 0.001]}  

    gridresults = GridSearchCV(SVC(), parameters, refit = True, verbose = 3) 
    gridresults.fit(X_train, Y_train.values.ravel()) 
    print(gridresults.best_params_) 
    print(gridresults.best_estimator_) 
    predictionsaftersearch = gridresults.predict(X_test) 
    acc = metrics.accuracy_score(prediction,Y_test)
    plt.plot(parameters['C'],acc)
    plt.legend()
    plt.title("SVC Kernels vs Accuaracy")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(parameters['gamma'],acc)
    plt.legend()
    plt.title("SVC Kernels vs Accuaracy")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.show()


    
    model = KNeighborsClassifier()
    params = {'n_neighbors':[5,8,10],
              'leaf_size':[1,2,3,5],
              'weights':['uniform', 'distance']}
    model1 = GridSearchCV(model, params)
    model1.fit(X_train,Y_train.values.ravel())
    print(model1.best_params_)
    prediction=model1.predict(X_test)
    print(metrics.accuracy_score(prediction,Y_test))
   

    modelfordtc= DecisionTreeClassifier(random_state=1234)
    params = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
              'random_state':[123]}
    modelwithgs = GridSearchCV(modelfordtc, param_grid=params, n_jobs=-1)
    modelwithgs.fit(X_train,Y_train)
    print(modelwithgs.best_params_)
    predictionfordtc=modelwithgs.predict(X_test)
    plt.plot(parameters['max_features'],acc)
    plt.legend()
    plt.title("Dtree vs Accuaracy")
    plt.xlabel("Dtree")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(parameters['min_samples_split'],acc)
    plt.legend()
    plt.title("Dtree vs Accuaracy")
    plt.xlabel("Dtree")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(parameters['min_samples_leaf'],acc)
    plt.legend()
    plt.title("Dtree vs Accuaracy")
    plt.xlabel("Dtree")
    plt.ylabel("Accuracy")
    plt.show()

"""
    Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""



    
