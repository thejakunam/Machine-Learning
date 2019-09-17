from DecisionTree import *
import random
import pandas as pd
from sklearn import model_selection

header = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps','deg-malig','breast','breast-quad','irradiat']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', header=None, names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps','deg-malig','breast','breast-quad','irradiat'])
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

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
t_pruned = prune_tree(t, [26, 11, 5])


print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t_pruned)
print("Accuracy on test = " + str(acc))

##Second Pruning Strategy
pruned_list=[]
while len(pruned_list) < 3:
  pruned_list.append(random.randint(1,maxID))
t_pruned2 = prune_tree(t,pruned_list)

for x in range(len(pruned_list)): 
    print(pruned_list[x])


print_tree(t_pruned2)


acc2 = computeAccuracy(test, t_pruned2)


print("Accuracy on test = " + str(acc2))
