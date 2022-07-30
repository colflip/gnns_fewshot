import csv
import datetime

import gat
import gcn
import graphsage

dataset = ['Cora', 'CiteSeer', 'Photo', 'cs', 'Computers', 'CoraFull']
spt = [1, 3, 5]

time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
f = open('output/gnns-fw-' + str(time_now) + '.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(['', 'models/shot/datasets', ] + dataset)

for i in spt:
    gcn_acc = ['gcn', i]
    for dataset_p in dataset:
        gcn_mean_var = gcn.main(dataset_p, i)
        gcn_acc.append(gcn_mean_var)
    csv_writer.writerow(gcn_acc)
for i in spt:
    gat_acc = ['gat', i]
    for dataset_p in dataset:
        gat_mean_var = gat.main(dataset_p, i)
        gat_acc.append(gat_mean_var)
    csv_writer.writerow(gat_acc)
for i in spt:
    sage_acc = ['sage', i]
    for dataset_p in dataset:
        sage_mean_var = graphsage.main(dataset_p, i)
        sage_acc.append(sage_mean_var)
    csv_writer.writerow(sage_acc)

f.close()
