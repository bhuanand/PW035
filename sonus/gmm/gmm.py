import numpy as np
import scipy as sp
import Pycluster
import os
import random

def muAndSigma(data, nclusters):
    return np.mean(data, axis = 0), np.cov(data, rowvar = 0), nclusters

class GaussianCluster(object):
    '''
    a class for holding values: (responsibilities) of
    each cluster in a Gaussian Model
    '''
    def __init__(self, mean, covariance, nClusters):
        '''
        constructor for GaussianCluster class
        input calues:
            mean, covariance.
        '''
        self._mean = mean
        self._covariance = covariance + 1e-3 * np.eye(len(covariance))
        self._covariance = np.diag(np.diag(self.covariance))
        self._nClusters = nClusters
        self._precisionMatrix = np.linalg.inv(self.covariance)
        self._determinant = np.fabs(np.linalg.det(self.covariance))
        self._denominator = ((2.0 * np.pi) ** (-self.nClusters / 2.0)) * (self.determinant ** -0.5)

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

    @property
    def nClusters(self):
        '''get the number of clusters'''
        return self._nClusters

    @property
    def precisionMatrix(self):
        '''get precisionMatrix'''
        return self._precisionMatrix
    
    @precisionMatrix.setter
    def precisionMatrix(self, iprecisionMatrix):
        '''set the precisionMatrix'''
        self._precisionMatrix = iprecisionMatrix

    @property
    def determinant(self):
        '''get the determinant'''
        return self._determinant
    
    @determinant.setter
    def determinant(self, ideterminant):
        '''set the determinant'''
        self._determinant = ideterminant

    @property
    def denominator(self):
        '''get the denominator'''
        return self._denominator

    @denominator.setter
    def denominator(self, idenominator):
        '''set the denominator'''
        self._denominator = idenominator

    def updateCluster(self, imu, isigma):
        '''update the mean and covariance values'''
        self.mean = imu
        self.covariance = isigma + 1e-3 * np.eye(len(isigma))

        # update the precision matrix and determinant
        self.precisionMatrix = np.linalg.inv(self.covariance)
        self.determinant = np.fabs(np.linalg.det(self.covariance))
        self.denominator = ((2.0 * np.pi) ** (-self.nClusters / 2.0)) * (self.determinant ** -0.5)

    def gaussianPDF(self, idata):
        '''
        applies Gaussian Probability Density Function
        to the input data.
        '''
        difference = idata - self.mean
        pdf = self.denominator * np.exp( -0.5 * np.dot( np.dot(difference.transpose(), self.precisionMatrix),  difference))
        return pdf
    
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
        
    @apriori.setter
    def apriori(self, iapriori):
        '''set the apriori vals'''
        self._apriori = iapriori

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
            models.append(GaussianCluster( *muAndSigma( self._data[i * chunkSize: (i + 1) * chunkSize], self.nClusters)))

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

                
        models = [GaussianCluster( *muAndSigma(randomData[i], self.nClusters)) for i in xrange(self._nClusters)]
        
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

        models = [GaussianCluster( *muAndSigma(clusterData[i], self.nClusters)) for i in xrange(self._nClusters)]

        apriori = np.ones(self._nClusters, dtype = np.float32) / np.array([len(elem) for elem in clusterData])

        return models, apriori
    

    def means(self):
        '''return the list of mean values of all comprising models'''
        res = [self.models[i].mean for i in xrange(self.nClusters)]

        return res

    def covariance(self):
        '''return the list covariance matrices of all comprising models'''
        res = [self.models[i].covariance for i in xrange(self.nClusters)]

        return res

    def loglikelihood(self, resp):
        '''compute the log likelihood of gaussian mixture'''
        return np.sum( np.log( np.sum( resp, axis = 1)))
    
    def expectationMaximization(self, iterations = 40):
        '''
        apply the expectation maximization algorithm to maximize the likelihood 
        of a data belonging to particular class. 
        '''
        likelihood = list()

        for i in xrange(iterations):
            # expectation step
            resp = self.eStep()
            
            likelihood.append(self.loglikelihood(resp))

            # check for convergence
            if i > 1 and abs(likelihood[-1] - likelihood[-2]) < 1e-1:
                break
            
            # maximization step
            self.mStep(resp)

    def eStep(self):
        '''expectation step'''
        resp = np.zeros((len(self.data), self.nClusters))
                        
        for i in xrange(len(self.data)):
            for j in xrange(self.nClusters):
                resp[i, j] = self.apriori[j] * self.models[j].gaussianPDF(self.data[i])
            
            resp[i] = resp[i] / np.sum(resp[i], axis = 0)
                
        return resp
        
    def mStep(self, resp):
        '''
        maximization step
        followed wikipedia
        http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Clustering/Expectation_Maximization_%28EM%29
        used formulae's on this page for computing.
        '''        
        # get the transpose of expected values for further usage
        # within the function
        respTranspose = resp.T

        # computing the mean matrix
        apriori_data_sum = np.dot(respTranspose, self.data)

        # maximize mean values, done :-)
        means = apriori_data_sum / respTranspose.sum(axis = 1)[:, np.newaxis]

        # computing the covariance matrix
        # 1. compute the difference term on numerator
        # 2. compute the whole term on numerator
        # 3. compute the covariance
        
        # 1. compute the difference term on numerator
        diffmatrix = np.zeros((len(self.data), self.nClusters))

        diffmatrix = [[data - mean for mean in self.means()] for data in self.data]

        diffmatrixsqr = np.multiply(diffmatrix, diffmatrix)

        # 2. compute the whole term on the numerator
        covariance = [np.dot(respTranspose[i], diffmatrixsqr[:, i]) for i in xrange(self.nClusters)]

        # 3. compute the covariance, done :-)
        covariance = covariance / respTranspose.sum(axis = 1)[:, np.newaxis]

        # now update the mean and covariance values of all models
        for i in xrange(self.nClusters):
            self.models[i].updateCluster(means[i], np.diag(covariance[i]))
            
        # now compute the probability of a data belonging to a particular cluster
        self.apriori = np.sum(respTranspose, axis = 1) / float(len(self.data))
