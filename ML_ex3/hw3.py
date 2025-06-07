import numpy as np
from typing import Union
import math

def poisson_log_pmf(k: Union[int, np.ndarray], rate: float) -> Union[float, np.ndarray]:
    """
    k: A discrete instance or an array of discrete instances
    rate: poisson rate parameter (lambda)

    return the log pmf value for instances k given the rate
    """

    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    log_rate = np.log(rate)
    log_factorial = np.log(np.vectorize(math.factorial)(k))

    log_p = -rate + k * log_rate - log_factorial
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean = np.mean(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    lower_bound = None
    upper_bound = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Critical value for (1-alpha) confidence interval
    z_critical = norm.ppf(1 - alpha/2)

    standard_error = np.sqrt(lambda_mle / n)
    
    margin_of_error = z_critical * standard_error

    # Calculate the confidence interval
    lower_bound = lambda_mle - margin_of_error
    upper_bound = lambda_mle + margin_of_error
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    samples = np.array(samples)
    rates = np.array(rates)

    # Pre compute constants
    sum_of_samples = np.sum(samples)
    num_samples = len(samples)
    log_factorial_sum = np.sum(np.log(np.vectorize(math.factorial)(samples)))

    # Calculate log-likelihood for each rate
    likelihoods = (sum_of_samples * np.log(rates) - rates * num_samples - log_factorial_sum)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.1,  # P(X=0, Y=0)
            (0, 1): 0.2,  # P(X=0, Y=1)
            (1, 0): 0.2,  # P(X=1, Y=0)
            (1, 1): 0.5   # P(X=1, Y=1)
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,  # P(X=0, C=0)
            (0, 1): 0.15,  # P(X=0, C=1)
            (1, 0): 0.35,  # P(X=1, C=0)
            (1, 1): 0.35   # P(X=1, C=1)
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.15,  # P(Y=0, C=0)
            (0, 1): 0.15,  # P(Y=0, C=1)
            (1, 0): 0.35,  # P(Y=1, C=0)
            (1, 1): 0.35   # P(Y=1, C=1)
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.05,   # P(X=0, Y=0, C=0)
            (0, 0, 1): 0.05,   # P(X=0, Y=0, C=1)
            (0, 1, 0): 0.10,   # P(X=0, Y=1, C=0)
            (0, 1, 1): 0.10,   # P(X=0, Y=1, C=1)
            (1, 0, 0): 0.10,   # P(X=1, Y=0, C=0)
            (1, 0, 1): 0.10,   # P(X=1, Y=0, C=1)
            (1, 1, 0): 0.25,   # P(X=1, Y=1, C=0)
            (1, 1, 1): 0.25,   # P(X=1, Y=1, C=1)
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndependent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Check if P(X,Y) = P(X) * P(Y) for all combinations
        # If not equal, then X and Y are dependent
        for x in [0, 1]:
            for y in [0, 1]:
                if abs(X_Y[(x, y)] - X[x] * Y[y]) > 1e-10:
                    return True
        
        return False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndependent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Check if P(X,Y|C) = P(X|C) * P(Y|C) for all combinations
        for c in [0, 1]:
            if C[c] > 0:  # Avoid division by zero
                for x in [0, 1]:
                    for y in [0, 1]:
                        prob_x_given_c = X_C[(x, c)] / C[c]
                        prob_y_given_c = Y_C[(y, c)] / C[c]
                        prob_xy_given_c = X_Y_C[(x, y, c)] / C[c]
                        
                        if abs(prob_xy_given_c - prob_x_given_c * prob_y_given_c) > 1e-10:
                            return False
        
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Calculate the normal PDF using the formula:
    # p(x) = (1 / (std * sqrt(2*pi))) * exp(-0.5 * ((x - mean) / std)^2)
    coefficient = 1 / (std * np.sqrt(2 * np.pi))
    exponent = - ((x - mean) / std * 2) ** 2
    p = coefficient * np.exp(exponent)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Store class value and dataset
        self.class_value = class_value
        self.dataset = dataset
        
        # Separate features from labels (last column contains class labels)
        self.features = dataset[:, :-1]
        self.labels = dataset[:, -1]
        
        # Calculate prior probability: P(Y=class_value)
        self.prior = np.mean(self.labels == class_value)
        
        # Filter samples that belong to this class
        class_mask = (self.labels == class_value)
        class_features = self.features[class_mask]
        
        # Store number of features
        self.num_features = class_features.shape[1]
        
        # Compute mean and standard deviation for each feature
        self.means = np.mean(class_features, axis=0)
        self.stds = np.std(class_features, axis=0)
        
        # Handle potential zero standard deviations 
        self.stds = np.maximum(self.stds, 1e-10)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.prior
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Initialize likelihood to 1 (we'll multiply probabilities)
        likelihood = 1.0
        
        # For each feature, calculate the normal PDF and multiply to get joint likelihood
        for i in range(self.num_features):
            feature_likelihood = normal_pdf(x[i], self.means[i], self.stds[i])
            likelihood *= feature_likelihood
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Joint probability = Likelihood * Prior
        # P(x, class) = P(x|class) * P(class)
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        joint_prob = likelihood * prior
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob



def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob
