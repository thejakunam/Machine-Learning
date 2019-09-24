import random
import pandas as pd
from sklearn import model_selection
import sys

#Input from user the location of data and the attributes

dataset= input("Input the dataset location i.e. URL of dataset")
header = list(input("Input Attribute names separated by a space: ").split())
df = pd.read_csv(dataset,header=None,names=header)

#Dataset 1 - IRIS
#header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

#Dataset 2 - BREAST CANCER
#header = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps','deg-malig','breast','breast-quad','irradiat']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', header=None, names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps','deg-malig','breast','breast-quad','irradiat'])

#Dataset 3 - CAR VALUATION
#header=['buying','maint','doors','persons','lug_boot','safety']
#df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',header=None,names=['buying','maint','doors','persons','lug_boot','safety'])

lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)
print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
maxID=0
for leaf in leaves:
    if leaf.id>maxID:
      maxID=leaf.id
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
    
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
    

trainDF, testDF = model_selection.train_test_split(df, test_size=0.3)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)

print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(train, t)
print("Accuracy on train = " + str(acc))
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

#TODO: You have to decide on pruning strategy
t_pruned = prune_tree(t,[25,11,5])



print("*************Tree after pruning  *******")
print_tree(t_pruned)
acc = computeAccuracy(test, t_pruned)
print("Accuracy on test = " + str(acc))


print("Strategy 1: Select n random nodes and prune them ")

size=4
while acc<1:
  old_acc=acc
  pruned_list=[]
  while len(pruned_list) < size and size<30:
    pruned_list.append(random.randint(2,maxID))
  t_pruned = prune_tree(t,pruned_list)
  acc = computeAccuracy(test, t_pruned)
  if acc<old_acc or size == 30:
    break
  size+=1



print("*************Tree after pruning Using Strategy1  *******")
print_tree(t_pruned)
acc = computeAccuracy(test, t_pruned)
print("Accuracy on test = " + str(acc))


print("Strategy 2: Prune all nodes above leaf ")

pruned_list=[]
leaves = getLeafNodes(t)
innerNodes = getInnerNodes(t)
for inner in innerNodes:
  id=inner.id
  if((2*id+1 in leaves) or (2*id+2 in leaves)):
    pruned_list.append(id)
    
t_pruned2 = prune_tree(t,pruned_list)

print("*************Tree after pruning Using Strategy2  *******")
print_tree(t_pruned2)
acc2=computeAccuracy(test,t_pruned2)
print("Accuracy on test = " + str(acc2))


print("Strategy 3: Prune all inner nodes with maximum depth  ")

innerNodes = getInnerNodes(t)
maxDepth = 0
for inner in innerNodes:
  if maxDepth < inner.depth:
    maxDepth = inner.depth
print("Max inner node Depth   " + str(maxDepth))

pruned_list=[]
for inner in innerNodes:
  if inner.depth == maxDepth:
    pruned_list.append(inner.id)
    
t_pruned3 = prune_tree(t,pruned_list)

print("*************Tree after pruning Using Strategy3  *******")
print_tree(t_pruned3)
acc3=computeAccuracy(test,t_pruned3)
print("Accuracy on test = " + str(acc3))
  
