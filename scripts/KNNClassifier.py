from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
from matplotlib.colors import ListedColormap
#from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D


class KNNClassifier():
    """
    KNNClassifier provides classification methods for Author Classification using K-Nearest Neighbor Algorithms
    """
    def __init__(self,K):
        #Setting default K
        self.K = K;
        self.weights = 'uniform';
        self.KNN = KNeighborsClassifier(n_neighbors = self.K,weights = self.weights);

    def Predict(self,X):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        #meshsize = .02;   #step size in mesh
        #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        #xx, yy = np.meshgrid(np.arange(x_min, x_max, meshsize),np.arange(y_min, y_max, meshsize))
        #YHat = self.KNN.predict(np.c_[xx.ravel(), yy.ravel()]);
        #self.DisplayClassification(X,YHat)
        return self.KNN.predict(X);


    def Train(self,phrases,authors):
        self.KNN.fit(phrases,authors);

    """
    Receiver Operator Characteristic
    """
    def ROC():None;

    def SetK(self,K):
        self.K = K;

    def SetClasses(self,classes):
        """
        Classes should be a list of Author IDs
        """
        self.classes = classes;

    def PredictSoft():None;

    def Classify():None;

    def GetK(self): return self.K;

    def GetClasses(self):None;

    def ShowPCA(self,X,Y):None;

    def DisplayClassification(self,X,YHat):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        meshsize = .02;   #step size in mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, meshsize),
                             np.arange(y_min, y_max, meshsize))

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Put the result into a color plot
        YHat = YHat.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, YHat, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=self.classes, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show();


    def DisplayFeatures(self,X,Y):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        plt.figure();
        plt.scatter(X[:,0],X[:,1],c = Y, cmap=plt.cm.Paired);
        plt.xlabel('Feature 1');
        plt.ylabel('Feature 2');

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

        plt.show()

    def ConfusionMatrix():None;

    def AreaUnderROC():None;


