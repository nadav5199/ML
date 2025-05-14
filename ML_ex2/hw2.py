import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # if the df is empty return 0
    if data.shape[0] == 0:
        return 0.0
    # count the labels
    labels = data[:,-1]
    # substract duplicates from the count
    _,count = np.unique(labels,return_counts=True)
    # save the frequency of each property
    frequency = count / len(labels)
    # calculate the gini value
    gini = 1 - np.sum(np.square(frequency)) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # same as in gini
    # if the df is empty return 0
    if data.shape[0] == 0:
        return 0.0
    # count the labels
    labels = data[:,-1]
    # substract duplicates from the count
    _,count = np.unique(labels,return_counts=True)
    # save the frequency of each property
    frequency = count / len(labels)
    # calculate the entropy
    entropy = np.sum([-freq * np.log2(freq) for freq in frequency if freq > 0])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # count the labels
        labels = self.data[:,-1]
        # check how many unique label and how many occurences for each
        unique_labels, count = np.unique(labels,return_counts=True)
        # save the label with the highest count
        pred = unique_labels[np.argmax(count)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # save the unique values of the given feature and store those rows in the dictionery
        unique_values = np.unique(self.data[:, feature])
        for val in unique_values:
            groups[val] = self.data[self.data[:,feature] == val] 
        
        # calculate the overall uncertainty
        overall_impurity = self.impurity_func(self.data)
        # calculate the weighted-sum of the children impurity
        size = len(self.data)
        if not self.gain_ratio:
            child_impurity = 0.0
            for val,subset in groups.items():
                weight = len(subset) / size
                child_impurity += weight * self.impurity_func(subset)
            goodness = overall_impurity - child_impurity
        else:
            split_info = 0.0
            for val,subset in groups.items():
                weight = len(subset) / size
                info_gain = calc_entropy(subset)
                split_info += -(weight) * np.log2(weight)
            if not split_info:
                goodness = 0.0
            else:
                goodness = info_gain / split_info
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        #  if leaf node or didn't split 
        if self.feature == -1:
            self.feature_importance = 0.0
            return
        gos,_ = self.goodness_of_split(self.feature)
        weight = len(self.data) / n_total_sample
        self.feature_importance = gos * weight
        return 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # edge cases
        # if we reached max depth
        if self.depth >= self.max_depth:
            self.terminal = True
            return
        # if nothing to split, set as leaf
        labels = self.data[:,-1]
        if len(np.unique(labels)) == 1:
            self.terminal = True
            return 
        # find the best split
        N = self.data.shape[1] - 1  # the amount of features in our DB without the labels
        best_feature, best_goodness = -1, -np.inf
        best_groups = None

        for i in range(N):
            goodness, group = self.goodness_of_split(i)
            if goodness > best_goodness:
                best_goodness = goodness
                best_groups = group
                best_feature = i
        # if dont have an improvment set as leaf
        if best_goodness <= 0 or best_feature == -1:
            self.terminal = True
            return 
        # post pruning
        if self.chi != 1:
            # compute X^2 
            unique_labels, counts = np.unique(labels,return_counts=True)
            labels_count = dict(zip(unique_labels,counts))

            chi = 0.0

            for subset in best_groups.values():
                sample_count = len(subset)
                for lbl, cnt in labels_count.items():
                    expected = (sample_count * cnt) / len(self.data)
                    observed = np.sum(subset[:,-1] == lbl)

                    if expected > 0:
                        chi += np.square(observed - expected) / expected

            deg_of_freedom = max(len(best_groups) - 1, 1)  # make sure it is not 0
            if deg_of_freedom in chi_table:
                crit = chi_table[deg_of_freedom][self.chi]
                if chi < crit:
                    self.terminal = True
                    return
            
        # create the children
        self.feature = best_feature
        for val, subset in best_groups.items():
            child = DecisionNode(data=subset, impurity_func=self.impurity_func, feature=-1, depth=self.depth + 1,
                                  chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, val)
            

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #
        
    # def depth(self):
    #     return self.root.depth

    def get_max_depth(self):
        if self.terminal or not self.children:
            return self.depth
        return max(self.get_max_depth(child) for child in self.children)

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Create the root node with the training data
        self.root = DecisionNode(
            data=self.data, 
            impurity_func=self.impurity_func, 
            depth=0, 
            chi=self.chi, 
            max_depth=self.max_depth, 
            gain_ratio=self.gain_ratio
        )
        
        # Total number of samples
        n_total_samples = len(self.data)

        def _build_recursive(node):
            node.split()
            node.calc_feature_importance(n_total_samples)

            if node.terminal:
                return
            
            for child in node.children:
                _build_recursive(child)

        _build_recursive(self.root)


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # init the node as the root 
        node = self.root

        while not node.terminal:
            feature_val = instance[node.feature] # the value of the feature in the given vector
            if feature_val in node.children_values:
                idx = node.children_values.index(feature_val)
                node = node.children[idx]
            else:
                break

        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if len(dataset) == 0:
            return 0
        
        correct = 0
        # on every row in the dataset compare the real value to the predicted value
        for r in dataset:
            observed = r[-1]
            predicted = self.predict(r)
            
            if observed == predicted:
                correct += 1

        accuracy = (correct * 100.0) / len(dataset) # change the corect values to accuracy
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        tree = DecisionTree(X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=False)
        tree.build_tree()

        training_acc = tree.calc_accuracy(X_train)
        validation_acc = tree.calc_accuracy(X_validation)

        training.append(training_acc)
        validation.append(validation_acc)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    def get_max_depth(node):
        if node.terminal:
            return node.depth
        
        max_depth = 0
        for child in node.children:
            depth = get_max_depth(child)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for chi in chi_values:
        tree = DecisionTree(X_train, impurity_func=calc_entropy, chi=chi, max_depth=10, gain_ratio=False)
        tree.build_tree()

        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(get_max_depth(tree.root))


    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






