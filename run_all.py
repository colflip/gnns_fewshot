import gat
import gcn
import graphsage

dataset = ['Cora', 'CiteSeer', 'Photo', 'cs', 'Computers']
spt = [1, 3, 5]
for dataset_p in dataset:
    for i in spt:
        gcn.main(dataset_p, i)
        gat.main(dataset_p, i)
        graphsage.main(dataset_p, i)
