from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import sqrt
from operator import itemgetter

moons = datasets.make_moons(n_samples = 1000, noise = 0.4)
set_k = 3
xtrain, xtest, ytrain, ytest = train_test_split(moons[0], moons[1])
results = []

#Distance between two points
def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

#Returns most common element in the list
def mode(lst):
    st = set(lst)
    N = [[x, lst.count(x)] for x in st]
    N.sort(key = itemgetter(1), reverse = True)
    #print(lst, N)
    return N.pop(0)[0]

#Finds distance between point and all train points,
#sorts by distance and returns a list of the indices of the k nearest points
def NN(point, k):
    S = []
    for trainindex in range(len(xtrain)):
        S.append([distance(point, xtrain[trainindex]), trainindex])
        S.sort()
    N_index = [i[1] for i in S[0:k]]
    return N_index

#Returns the number of incorrectly labeled moons
def score(a, b):
    sum = 0
    for i in range(len(a)):
        sum += abs(a[i] - b[i])
    return sum

#Takes the list of indices from NN and
#finds the label from ytrain corresponding to the closest points in xtrain and
#finds the most common label in that subset of ytrain
for y in xtest:
    NN_indices = NN(y, set_k)
    l = [ytrain[i] for i in NN_indices]
    results.append(mode(l))

#Change the number incorrectly labeled moons and converts to percentage
print('percent right:', 100 - (score(results, ytest)/len(ytest) * 100), '%')


