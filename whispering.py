from collections import Counter

from random import randint
from collections import Counter

def whispering(graph, reset=True):
    '''
    Takes in an initial graph and modifies its predicted labels by performinig
    the Chinese whispering algorithm
    
    INPUTS:
    
        graph:  The graph object returned by populate_graph()
                tuple(list of lists len N, list len N, list len N)
        
        reset:  if True, the method resets any previous clustering performed on the graph
                if False, the method starts from the end of the previous run
                bool
    
    OUTPUT:
    
        graph:  The same graph object
    '''
    edges, truth, predicted = graph
    N = len(edges)
    assert N == len(truth)
    assert N == len(predicted)
    
    # reset if needed
    if reset:
        predicted = list(i for i in range(N))
    
    remaining = Counter(predicted)
    since = 0    # iterations since last class deletion
    while(since < N):
        # pick random node
        node = randint(0, N-1)
        old_label = predicted[node]
        
        # count up all the labels of its neighbors
        count = Counter()
        for neighbor, weight in edges[node]:
            count[predicted[neighbor]] += weight
        new_label, _ = count.most_common(1)[0]
        
        # change the old label to a new one
        predicted[node] = new_label
        
        # check if class was removed
        since += 1
        count = remaining[old_label]
        if count <= 1:
            del remaining[old_label]
            since = 0
        else:
            remaining[old_label] = count - 1
        remaining[new_label] += 1
        
        # break out of algorithm if plateaued
    return graph

def __matches(primary, secondary):
    '''
    Seperates primary into groups
    For every pair within groups, calculate if secondaries match
    '''
    # separate into groups
    groups = {}
    for i, x in enumerate(primary):
        if x not in groups:
            groups[x] = [i]
        else:
            groups[x].append(i)
    
    # iterate internal pairs
    match = 0
    not_match = 0
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                if secondary[group[i]] == secondary[group[j]]:
                    match += 1
                else:
                    not_match += 1
    return match, not_match
    

def metrics(graph):
    '''
    Analyzes the given graph and returns pertinent performance metrics
    
    INPUT:
    
        graph:      The graph object returned by populate_graph()
                    tuple(list of lists len N, list len N, list len N)
    
    OUTPUT:
    
        precision:  Metric worsened by false negatives
                    float in range [0, 1]
        
        recall:     Metric worsened by false positives
                    float in range [0, 1]
    '''
    
    edges, truth, predicted = graph
    tt, ft = __matches(truth, predicted)
    _, tf = __matches(predicted, truth)
    
    precision = tt/(tt+ft)
    recall = tt/(tt+tf)
    
    return precision, recall
