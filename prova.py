from pyspark import SparkContext, SparkConf
import sys
import os
import random as random
import csv



def belong_cell(doc):
    i=int(doc[0]/side)
    j=int(doc[1]/side)
    return ((i,j),)


def cell_count_per_doc(doc):
    pairs_dict = {}
    for p in doc:
        if p not in pairs_dict.keys():
            pairs_dict[p] = 1
        else:
            pairs_dict[p] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]



def points_count_with_partition(doc):
	points_count = (doc.flatMap(belong_cell) # <-- MAP PHASE (R1)
		.mapPartitions(cell_count_per_doc)    # <-- REDUCE PHASE (R1)
		.groupByKey()                              # <-- SHUFFLE+GROUPING
		.mapValues(lambda vals: sum(vals)))        # <-- REDUCE PHASE (R2)
	return points_count

# CHECKING NUMBER OF CMD LINE PARAMTERS
assert len(sys.argv) == 6, "Usage: python prova.py <D> <M> <K> <L> <file_name>" 

# SPARK SETUP
conf = SparkConf().setAppName('prova')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

# INPUT READING

# 1. Read number of partitions
L=sys.argv[4]
assert L.isdigit(), "L must be an integer"
L = int(L)
# SETTING AND PRINTING GLOBAL VARIABLES
D=sys.argv[1]
D = float(D)
M=sys.argv[2]
K=sys.argv[3]
side=float(D)/(2*(2**(1/2)))
#side = 1

# 2. Read input file and subdivide it into L random partitions
data_path = sys.argv[5]
assert os.path.isfile(data_path), "File or folder not found"
rdd = sc.textFile(data_path).repartition(numPartitions=L)
t_rdd = rdd.map(lambda line: line.split(","))
doc = t_rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()



def tipo(doc):
    return[cell for cell in doc.collect()][:20]#da completare


#print(tipo(doc))
cell = points_count_with_partition(doc)
num_cell = cell.count()
print("Output:", num_cell)
#print(doc.collect())

