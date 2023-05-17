# %%
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
from random import sample

# %%
class DecisionNode():
    '''
    Constructor for decision node
    attr - attribute to be tested on
    thresh - threshold for numerical attr, if categorical set to none
    children - dict mapping attr value to child, if numeric child keys are 'le' and 'gr'
    label - class label. default None, if not none, indicates that this is a leaf
    '''
    def __init__(self, attr, children, label=None, thresh=None):
        self.attr = attr
        self.children = children
        self.label = label
        self.thresh = thresh

def learn_tree(data,attr_list,do_ig,attr_vals,targets,do_forest=True,min_size_split=0,min_gain=0.0,max_depth=-1,maj_prop=1.0):
    '''
    Function to create decision tree based on data
    do_ig - boolean, True to use info gain, False to use gini critereon
    attr_vals - dict, maps to array of possible values for each attr, empty if numeric
    targets - array, possible values for target class
    min_size_split - int, min number of instances required in data before returning a leaf node
    min_gain - float, min info gain to determine when to stop
    max_depth - int, if -1 no depth limit, else maximum depth before stopping 
    '''
    #if num data too small, return leaf
    if len(data) <= min_size_split:
        return DecisionNode(None,None,data['class'].mode()[0],None)
    #if max depth reached, return leaf
    if max_depth == 0:
        return DecisionNode(None,None,data['class'].mode()[0],None)

    if (data['class'].value_counts().iloc[0]/len(data)) >= maj_prop:
        return DecisionNode(None,None,data['class'].mode()[0],None)

    #find best attr from list using info gain
    max_gain = float('-inf')
    min_gini = float('inf')
    best_attr = ''

    random_attr = sample(attr_list,math.ceil(math.sqrt(len(attr_list)))) if do_forest else attr_list

    for attr in random_attr:
        ig_attr,thresh = get_info_gain(data,attr,attr_vals[attr],targets)
        if ig_attr > max_gain:
            max_gain = ig_attr
            best_attr = attr
            best_thresh = thresh

    if best_attr == '':
        return DecisionNode(None,None,data['class'].mode()[0],None)

    best_vals = attr_vals[best_attr]
    best_children = defaultdict(lambda: DecisionNode(None,None,data['class'].mode()[0],None))
    
    #if numeric attr
    if len(best_vals) == 0:
        best_children['le'] = learn_tree(data.loc[data[best_attr] <= best_thresh],attr_list,do_ig,attr_vals,targets,do_forest,min_size_split,max_gain,max_depth-1,maj_prop) if len(data.loc[data[best_attr] <= best_thresh]) > 0 else DecisionNode(None,None,data['class'].mode()[0],None)
        best_children['gr'] = learn_tree(data.loc[data[best_attr] > best_thresh],attr_list,do_ig,attr_vals,targets,do_forest,min_size_split,max_gain,max_depth-1,maj_prop) if len(data.loc[data[best_attr] > best_thresh]) > 0 else DecisionNode(None,None,data['class'].mode()[0],None)
    
    else:
        for i in best_vals:
            next_child = learn_tree(data.loc[data[best_attr] == i],attr_list,do_ig,attr_vals,targets,do_forest,min_size_split,max_gain,max_depth-1,maj_prop) if len(data.loc[data[best_attr] == i]) > 0 else DecisionNode(None,None,data['class'].mode()[0],None)
            best_children[i] = next_child
            
    return DecisionNode(best_attr,best_children,thresh=best_thresh)


#Helper functions for decision tree
def get_info_gain(data,attr,num_cat,targets):
    split_entropy = 0
    if len(num_cat) > 0:
        best_split = None
        for i in num_cat:
            split_data = data.loc[data[attr] == i]
            split_entropy += (entropy(split_data,targets)*len(split_data))/len(data)
    else:
        best_split = data[attr].mean()
        split_entropy += split_thresh(data,attr,best_split,targets)

    return (entropy(data,targets) - split_entropy),best_split

#determine split threshhold for numeric attr
def split_thresh(data,attr,thresh,targets):
    if math.isnan(thresh):
        return entropy(data,targets)
    df_le = data.loc[data[attr] <= thresh]
    df_g = data.loc[data[attr] > thresh]
    return (entropy(df_le,targets)*len(df_le) + entropy(df_g,targets)*len(df_g))/len(data)

def entropy(data,targets):
    if len(data) == 0:
        return 0
    ent = 0
    for i in range(len(targets)):
        try:
            prob_i = data['class'].value_counts().iloc[i]/len(data)
            ent -= (prob_i * math.log(prob_i,2))
        except:
            pass
    return ent

#function to classify an instance based on decision tree
def classify(instance,node):
    while node.label is None:
        if node.thresh is None:
            node = node.children[instance[node.attr]]
        else:
            node = node.children['le'] if instance[node.attr] <= node.thresh else node.children['gr']
    guess = node.label
    return guess

#wrapper fucntion for classify which takes all trees in forest, takes majority output and returns that guess
def ntree_classify(instance,ntree):
    votes = []
    for tree in ntree:
        votes.append(classify(instance,tree))
    count = Counter(votes)
    return count.most_common(1)[0][0]

#test accuracy on test data
def test_decision(to_test,targs,ntrees):
    predictions = pd.DataFrame(to_test.apply(lambda row: ntree_classify(row,ntrees), axis=1),columns = ['class'])
    predictions['actual'] = to_test.loc[predictions.index,['class']]
    prec,rec,f1 = [0,0,0]

    for val in targs:
        is_targ = predictions[predictions['class'] == val]
        not_targ = predictions[predictions['class'] != val]
        tp = len(is_targ[is_targ['class'] == is_targ['actual']])
        fp = len(is_targ[is_targ['class'] != is_targ['actual']])
        fn = len(not_targ[not_targ['actual'] == val])
        tn = len(not_targ[not_targ['actual'] != val])
        this_prec = (tp/(tp+fp)) if (tp+fp) > 0 else 0
        this_rec = (tp/(tp+fn)) if (tp+fn) > 0 else 0
        f1 += (this_prec*this_rec*2)/(this_rec+this_prec) if (this_rec+this_prec) > 0 else 0
        prec += this_prec
        rec += this_rec

    avg_prec = prec/len(targs)
    avg_rec = rec/len(targs)
    avg_f1 = f1/len(targs)
    accuracy = len(predictions[predictions['class'] == predictions['actual']])/len(to_test)
    return [accuracy,avg_prec,avg_rec,avg_f1]

np.random.seed(1)
k = 10
#function to do cross fold validation
def k_fold(fold,attr_list,attr_vals,targets,nvals,do_forest = True, max_depth=7,min_size_split=10,maj_prop=0.9):
    #maps n vals to list of average statistics for each 
    fold_metrics = defaultdict(list)
    #iterate through folds, taking turns being test fold
    for i in range(k):
        test_fold = fold[i]
        train_fold = fold[0:i]
        train_fold.extend(fold[i+1:len(fold)])
        train_data = pd.concat(train_fold)
        #vary number of trees for this fold
        for n in nvals:
            #list to store decisions of each tree by index
            ntree = []
            for j in range(n):
                #generate bootstrap w replacment
                bootstrap = train_data.sample(frac=1,replace=True)
                ntree.append(learn_tree(bootstrap,attr_list,True,attr_vals,targets,do_forest,max_depth=max_depth,min_size_split=min_size_split,maj_prop=maj_prop))
            fold_metrics[n].append(test_decision(test_fold,targets,ntree))
    return fold_metrics