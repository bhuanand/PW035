import numpy as np
import scipy as sp
import Pycluster
import os
import random

def muAndSigma(data):
    return np.mean(data, axis = 0), np.cov(data, rowvar = 0)

class GaussianCluster(object):
    '''
    a class for holding values: (responsibilities) of
    each cluster in a Gaussian Model
    '''
    def __init__(self, mean, covariance):
        '''
        constructor for GaussianCluster class
        input calues:
            mean, covariance.
        '''
        self._mean = mean
        self._covariance = covariance

    @property
    def mean(self):
        '''get the mean value'''
        return self._mean

    @mean.setter
    def mean(self, imean):
        '''set the mean value'''
        self._mean = imean

    @property
    def covariance(self):
        '''get the covariance value'''
        return self._covariance
    
    @covariance.setter
    def covariance(self, icovariance):
        '''set the covariance value'''
        self._covariance = icovariance
    
class GaussianMixtureModel(object):
    '''
    representation of Gaussian Mixture Model probability distribution.
    
    initializes the parameters such that every mixture component has means
    calculated from K - means cluster algorithm and covariance type is diagonal.

    Expectation Maximization:
        Gaussian Mixture Models are typically fitted using EM algorithm.
        this algorithm makes iterative estimation of model parameters(thetas)
        for the GMM's.
        
        the implementation of EM algorithm starts with the initial estimations
        for mu and sigma (mean and covariance).

        algorithm iteratively improve the values of mean and covariance
        using two steps:
            E Step: (Expectation Step)
            M Step: (Maximization Step)
    '''
    def __init__(self, data, nClusters, options = {}):
        self._data = data
        self._nClusters = nClusters
        self._options = options
        self._models, self._apriori = self.initializeClusters()
        
    @property
    def data(self):
        '''get the data'''
        return self._data
    
    @property
    def nClusters(self):
        '''get number of clusters'''
        return self._nClusters

    @property
    def options(self):
        '''get the properties'''
        return self._options

    @property
    def models(self):
        '''get the models in gmm'''
        return self._models

    @property
    def apriori(self):
        '''get the apriori values'''
        return self._apriori
        
    def initializeClusters(self):
        '''
        Given the number of clusters and the data,
        lets initialize the responsibilities of each class.
        
        responsibilities include 
        '''
        # get the dimension of the input data
        rows, cols = self._data.shape

        # lets assume the data is X = {xi, xi+1, xi+2,... , xn}
        # where each of xi is a D - dimensional.

        # now initialize the values for \mu and \sigma
        # representing the mean and covariance for each of the clusters
        
        models = []
        aproiri = []

        if not self._options == dict():
            # check if the options are not specified

            if self._options.get('method', False):
                # get the method of initialization specified
                
                method = self._options.get('method')
                
                if method == 'uniform':
                    models, apriori = self.__uniform_initialization()
                elif method == 'random':
                    models, apriori = self.__random_initialization()
                else:
                    models, apriori = self.__kmeans_initialization()
            else:
                models, apriori = self.__kmeans_initialization()
        else:
            models, apriori = self.__kmeans_initialization()
        
        return models, apriori

    def __uniform_initialization(self):
        '''
        given the data points, uniformly assign them to different 
        clusters then estimate the parameters
        '''
        
        # shuffle the data in the input
        np.random.shuffle(self._data)

        rows, cols = self._data.shape

        chunkSize = rows / self._nClusters

        # create clusters with the data got from chunks
        models = []
        for i in xrange(self._nClusters):
            models.append(GaussianCluster( *muAndSigma( self._data[i * chunkSize: (i + 1) * chunkSize])))

        apriori = np.ones(self._nClusters, dtype = np.float32) / self._nClusters
        
        return models, apriori

    def __random_initialization(self):
        ''' 
        given the data points, randomly assign them to different
        clsuters then estimate the parameters
        '''
        
        # set the seed value for the random
        random.seed(os.getpid())

        # now select random samples from the data
        randomSamples = random.sample(self._data, self._nClusters)
        
        # temporary list to hold the data for each clusters
        randomData = [[] for i in xrange(self._nClusters)]

        # assign the data to clusters
        for vector in self._data:
            index = np.argmin( [np.linalg.norm(vector - row) for row in randomSamples])
            
            if index < self._nClusters:
                randomData[index].append(vector) 
            else:
                randomData[random.randint(0, self._nClusters - 1)].append(vector)

                
        models = [GaussianCluster( *muAndSigma(randomData[i])) for i in xrange(self._nClusters)]
        
        apriori = np.ones(self._nClusters, dtype = np.float32) / np.array([len(elem) for elem in randomData])

        return models, apriori

    def __kmeans_initialization(self):
        '''
        given the data points, cluster them by applying kmeans clustering
        algorithm.
        '''
        # apply kmeans clustering to get the centroids and labels for each vector in data
        labels, error, nfound = Pycluster.kcluster(self._data, self._nClusters)

        clusterData = [[] for i in xrange(self._nClusters)]

        # assign vectors to clusters
        for data, label in zip(self._data, labels):
            clusterData[label].append(data)

        models = [GaussianCluster( *muAndSigma(clusterData[i])) for i in xrange(self._nClusters)]

        apriori = np.ones(self._nClusters, dtype = np.float32) / np.array([len(elem) for elem in clusterData])

        return models, apriori
