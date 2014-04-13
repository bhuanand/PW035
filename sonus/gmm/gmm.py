import numpy as np
import scipy as sp
import Pycluster
import os
import sys
import random
import cPickle

def muAndSigma(data, nclusters):
    """
    compute mean and covariance for each cluster.

    :param data: data for the cluster found by using initialization.
    :param nclusters: number of clusters
    :return: returns a tuple consisting of mean and covariance.
    """
    return np.mean(data, axis = 0), np.cov(data, rowvar = 0), nclusters

class GaussianCluster(object):
    """
    a class for holding values: (responsibilities) of
    each cluster in a Gaussian Model
    """
    def __init__(self, mean, covariance, nClusters):
        """
        constructor for GaussianCluster class
        :param mean: mean for clsuter
        :param covariance: covariance for the cluster
        :param nClusters: number of clusters
        """
        self._mean = mean
        self._covariance = covariance + 1e-3 * np.eye(len(covariance))
        self._covariance = np.diag(np.diag(self.covariance))
        self._nClusters = nClusters
        self._precisionMatrix = np.linalg.inv(self.covariance)
        self._determinant = np.fabs(np.linalg.det(self.covariance))
        self._denominator = ((2.0 * np.pi) ** (-self.nClusters / 2.0)) * (self.determinant ** -0.5)

    @property
    def mean(self):
        """get the mean value"""
        return self._mean

    @mean.setter
    def mean(self, imean):
        """set the mean value"""
        self._mean = imean

    @property
    def covariance(self):
        """get the covariance value"""
        return self._covariance

    @covariance.setter
    def covariance(self, icovariance):
        """set the covariance value"""
        self._covariance = icovariance

    @property
    def nClusters(self):
        """get the number of clusters"""
        return self._nClusters

    @property
    def precisionMatrix(self):
        """get precisionMatrix"""
        return self._precisionMatrix

    @precisionMatrix.setter
    def precisionMatrix(self, iprecisionMatrix):
        """set the precisionMatrix"""
        self._precisionMatrix = iprecisionMatrix

    @property
    def determinant(self):
        """get the determinant"""
        return self._determinant

    @determinant.setter
    def determinant(self, ideterminant):
        """set the determinant"""
        self._determinant = ideterminant

    @property
    def denominator(self):
        """get the denominator"""
        return self._denominator

    @denominator.setter
    def denominator(self, idenominator):
        """set the denominator"""
        self._denominator = idenominator

    def updateCluster(self, imu, isigma):
        """
        update the mean and covariance values
        :param imu: new mean for cluster
        :param isigma: new covariance for cluster
        """
        self.mean = imu
        self.covariance = isigma + 1e-3 * np.eye(len(isigma))

        # update the precision matrix and determinant
        self.precisionMatrix = np.linalg.inv(self.covariance)
        self.determinant = np.fabs(np.linalg.det(self.covariance))
        self.denominator = ((2.0 * np.pi) ** (-self.nClusters / 2.0)) * (self.determinant ** -0.5)

    def gaussianPDF(self, idata):
        """
        applies Gaussian Probability Density Function
        to the input data.
        :param idata: data vector for computing density
        """
        difference = idata - self.mean
        pdf = self.denominator * np.exp( -0.5 * np.dot( np.dot(difference.transpose(), self.precisionMatrix),  difference))
        return pdf

class GaussianMixtureModel(object):
    """
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
    """
    def __init__(self, data, nClusters, options = {}):
        """
        constructor for GaussianMixtureModel class
        :param data: mixture or data to be clustered
        :param nClusters: number of clusters
        :param options: options such os specifying method of initialization
        """
        self._data = data
        self._nClusters = nClusters
        self._options = options
        self._models, self._apriori = self.initializeClusters()

    @property
    def data(self):
        """get the data"""
        return self._data

    @property
    def nClusters(self):
        """get number of clusters"""
        return self._nClusters

    @property
    def options(self):
        """get the properties"""
        return self._options

    @property
    def models(self):
        """get the models in gmm"""
        return self._models

    @property
    def apriori(self):
        """get the apriori values"""
        return self._apriori

    @apriori.setter
    def apriori(self, iapriori):
        """set the apriori vals"""
        self._apriori = iapriori

    def initializeClusters(self):
        """
        Given the number of clusters and the data,
        lets initialize the responsibilities of each class.

        responsibilities include
        """
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

                if method == 'kmeans':
                    models, apriori = self.__kmeans_initialization()
                elif method == 'random':
                    models, apriori = self.__random_initialization()
                else:
                    models, apriori = self.__uniform_initialization()
            else:
                models, apriori = self.__uniform_initialization()
        else:
            models, apriori = self.__uniform_initialization()

        return models, apriori

    def __uniform_initialization(self):
        """
        given the data points, uniformly assign them to different
        clusters then estimate the parameters
        """

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
        """
        given the data points, randomly assign them to different
        clsuters then estimate the parameters
        """

        # set the seed value for the random
        random.seed(os.getpid())

        # now select random samples from the data
        randomSamples = random.sample(self._data, self._nClusters)

        # temporary list to hold the data for each clusters
        randomData = [[] for i in xrange(self._nClusters)]

        # assign the data to clusters
        for vector in self._data:
            randomData[random.randint(0, self._nClusters - 1)].append(vector)


        models = [GaussianCluster( *muAndSigma(randomData[i], self.nClusters)) for i in xrange(self._nClusters)]

        apriori = np.ones(self._nClusters, dtype = np.float32) / np.array([len(elem) for elem in randomData])

        return models, apriori

    def __kmeans_initialization(self):
        """
        given the data points, cluster them by applying kmeans clustering
        algorithm.
       """
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
        """return the list of mean values of all comprising models"""
        res = [self.models[i].mean for i in xrange(self.nClusters)]

        return res

    def covariance(self):
        """return the list covariance matrices of all comprising models"""
        res = [self.models[i].covariance for i in xrange(self.nClusters)]

        return res

    def loglikelihood(self, resp):
        """compute the log likelihood of gaussian mixture
        :param resp: resp matrix computed by applying E step of em algorithm
        """
        return np.sum( np.log( np.sum( resp, axis = 1)))

    def fit(self, data):
        """fits the data GMM, and return the label of GMM model it belongs
        :param data: fit the data against the model

        returns the cluster it belongs
        """
        resp = self.eStep(data=data)

        res = resp.argmax(axis = 1)

        unique_vals, indices = np.unique(res, return_inverse=True)

        return unique_vals[ np.argmax( np.bincount( indices))]

    def expectationMaximization(self, iterations = 10):
        """
        apply the expectation maximization algorithm to maximize the likelihood
        of a data belonging to particular class.
        :param iterations: maximum iterations
        """
        likelihood = list()

        for i in xrange(iterations):
            # expectation step
            resp = self.eStep()

            likelihood.append(self.loglikelihood(resp))

            # check for convergence
            if i > 1 and abs(likelihood[-1] - likelihood[-2]) < abs(1e-4):
                break

            # maximization step
            self.mStep(resp)

    def eStep(self, data = None):
        """expectation step
        :param data: input data
        """
        if data == None:
            data = self.data
        else:
            data = data
        resp = np.zeros((len(data), self.nClusters))

        for i in xrange(len(data)):
            for j in xrange(self.nClusters):
                resp[i, j] = self.apriori[j] * self.models[j].gaussianPDF(data[i])

            resp[i] = resp[i] / np.sum(resp[i], axis = 0)

        return resp

    def mStep(self, resp):
        """
        maximization step
        followed wikipedia
        http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Clustering/Expectation_Maximization_%28EM%29
        used formulae's on this page for computing.
        """
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

        diffmatrix = [[d - mean for mean in self.means()] for d in self.data]

        diffmatrixsqr = np.multiply(diffmatrix, diffmatrix)

        # 2. compute the whole term on the numerator
        covariance = [np.dot(respTranspose[i], diffmatrixsqr[:, i]) for i in xrange(self.nClusters)]

        # 3. compute the covariance, done :-)
        covariance = covariance / respTranspose.sum(axis = 1)[:, np.newaxis]

        # now update the mean and covariance values of all models
        for i in xrange(self.nClusters):
            self.models[i].updateCluster(means[i], np.diag(covariance[i]))

        # now compute the probability of a self.data belonging to a particular cluster
        self.apriori = np.sum(respTranspose, axis = 1) / float(len(self.data))


    @classmethod
    def saveobject(cls, obj, filepath = None):
        """saves the gmm object for next time usage
        :param obj: object to be stored
        :param filepath: file path at which the object is to be stored
        """

        homedir = os.path.expanduser('~')

        sonus_gmmobject = os.path.join('sonus', 'gmm-object')

        # if file name is not specified
        # store the object at the ~/sonus/gmm-object
        if not filepath:

            # check if the path already exists
            if os.path.exists(os.path.join(homedir, sonus_gmmobject)):

                # store the object to this file
                fobj = open(os.path.join(homedir, sonus_gmmobject), 'wb')
            else:
                # check if default path already exists
                if not os.path.exists(os.path.join(homedir, 'sonus')):

                    # create the directory and files then store the object to the file
                    os.mkdir( os.path.join( homedir, 'sonus'))

                # create the file
                fobj = open( os.path.join( homedir, sonus_gmmobject), 'wb')

            # write to the file
            cPickle.dump( obj, fobj)

            print 'saved object to file: {0}'.format(fobj.name)

            fobj.close()
        else:
            # file name is specified

            filepath = os.path.expanduser(filepath)

            # check if the specified path exists
            if os.path.exists(filepath):

                # check if it is a filepath
                if not os.path.isfile(filepath):

                    # check if it is a directory
                    if os.path.isdir(filepath):

                        # create a file under this directory
                        fobj = open(os.path.join(filepath, 'gmm-object'), 'wb')
                else:
                    # it is a file path, open it
                    fobj = open(filepath, 'wb')
            else:
                # create file at default path ~/sonus/
                if not os.path.exists(os.path.join(homedir, 'sonus')):

                    # create the directory and files then store the object to the file
                    os.mkdir( os.path.join( homedir, 'sonus'))

                # create the file
                fobj = open( os.path.join( homedir, sonus_gmmobject), 'wb')

            # write to the file
            cPickle.dump( obj, fobj)

            print 'saved to the file: {0}'.format(fobj.name)

            fobj.close()

    @classmethod
    def loadobject(cls, filepath = None):
        """loads the previosly stored object
        :param filepath: file path to load the object from
        """

        homedir = os.path.expanduser('~')

        sonus_gmmobject = os.path.join('sonus', 'gmm-object')

        obj = None

        # file path is not specified
        # try to load it from default path ~/sonus/gmm-object
        if not filepath:

            # if the ~/sonus/gmm-object already exists
            if os.path.exists(os.path.join(homedir, sonus_gmmobject)):
                fobj = open(os.path.join(homedir, sonus_gmmobject), 'rb')

                obj = cPickle.load(fobj)
            else:
                #if the ~/sonus/gmm-object file doesnt exists, raise error

                errormsg = '''
                file path argument is not specified.\n
                tried loading from path {0}. but the path {0} doesnt exists.\n
                please either specify a valid filepath or make sure you have\n
                saved the object.
                '''.format(os.path.join(homedir, sonus_gmmobject))

                raise Exception(errormsg)
        else:
            # user has specified the file path

            # expand ~ if present
            filepath = os.path.expanduser(filepath)

            # check if the path exists
            if os.path.exists(filepath):

                # check if it is a file
                if not os.path.isfile(filepath):

                    # check if it is a directory
                    if os.path.isdir(filepath):

                        # raise error
                        raise Exception('please specify valid file path, you have specified a directory')
                else:
                    try:
                        # it is a file path, open it
                        fobj = open(filepath, 'rb')

                        obj = cPickle.load(fobj)
                    except EOFError, e:
                        print 'the file doesnt contain valid object.' + e.message
            else:
                # user specified file doesnt exists

                # check if default path for stroring object exists
                if os.path.exists(os.path.join(homedir, sonus_gmmobject)):

                    # store the object
                    fobj = open(os.path.join(homedir, sonus_gmmobject), 'rb')

                    obj = cPickle.load(fobj)
                else:
                    # default path for storing object doesnt exists
                    errormsg = '''
                    the file path specified: {0} doesnt exists.\n
                    tried loading the object from default location {1}.\n
                    but the default path also doesnt exist. please make sure to\n
                    give the valid file name or to call saveobject before loading it.
                    '''.format(filepath, os.path.join(homedir, sonus_gmmobject))

                    raise Exception(errormsg)

        # check if the loaded object is an instance of this class
        if not isinstance(obj, cls):
            errormsg = '''
            the file doesnt contain the object of type {0}.
            '''.format(cls.__name__)

            raise Exception(errormsg)

        print 'loading from the file: {0}'.format(fobj.name)

        fobj.close()

        return obj

__all__ = [GaussianMixtureModel]
