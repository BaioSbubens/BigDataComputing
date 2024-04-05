from pyspark import SparkContext, SparkConf
import sys
import os
import random as random
import math

def distance(point1, point2):
    return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2

def exactOutliers(points, D, M, K):
    l=[]
    num_outliers=0
    for i in points:
        s=0
        for j in points:
            d=distance(i,j)
            if d<D**2:
                s+=1
        if s<=M:
            num_outliers+=1
            l.append(i)
    print("Number of outliers:", num_outliers)
    for k in sorted(l)[:K]:
        print(f'Point: {k}')


def belong_cell(doc):
    i=math.floor((doc[0]/side))
    j=math.floor((doc[1]/side))
    return ((i,j),)


def cell_count_per_doc(doc):
    pairs_dict = {}
    for p in doc:
        if p not in pairs_dict.keys():
            pairs_dict[p] = 1
        else:
            pairs_dict[p] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]



def step_A(doc):
	points_count = (doc.flatMap(belong_cell) # <-- MAP PHASE (R1)
		.mapPartitions(cell_count_per_doc)    # <-- REDUCE PHASE (R1)
		.groupByKey()                              # <-- SHUFFLE+GROUPING
		.mapValues(lambda vals: sum(vals)))        # <-- REDUCE PHASE (R2)
	return points_count

def step_B(cells):
    outliers = 0
    uncertain = 0
    for center in cells:
        i,j = center[0]
        tot_3x3 = 0
        tot_7x7 = 0
        for cell in cells:
            x,y = cell[0]
            d =math.sqrt(((i-x)**2 + (j-y)**2))
            if d <= diag:
                tot_3x3 += cell[1]
                #tot_7x7 += cell[1]
            elif d <= 3*diag:
                tot_7x7 += cell[1]
            if tot_3x3 > M:
                break
        tot_7x7 += tot_3x3
        if tot_7x7 <= M:
            outliers += center[1]
        elif (tot_3x3 <= M) and (tot_7x7 > M):
            uncertain += center[1]
    return (outliers, uncertain)

def first_k_cells(output_A):
    ordered = (output_A.map(lambda x: (x[1],x[0]))
               .sortByKey()
               .take(K))
    return ordered

def MRAApproxOutliers(point, D, M, K):
    output_A= step_A(point)
    cells = step_A(point).collect()
    outliers, uncertain = step_B(cells)
    print(f'Number of sure outliers: {outliers}')
    print(f'Number of uncertain outliers: {uncertain}')
    for el in first_k_cells(output_A):
        print(f'cell : {el[1]}, size: {el[0]}')

# CHECKING NUMBER OF CMD LINE PARAMTERS
assert len(sys.argv) == 6, "Usage: python provagruppo.py <D> <M> <K> <L> <file_name>" 

# SPARK SETUP
conf = SparkConf().setAppName('provagruppo')
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
M = int(M)
K=sys.argv[3]
K = int(K)

side=D/(2*math.sqrt(2))
diag = math.sqrt(2)

# 2. Read input file and subdivide it into L random partitions
data_path = sys.argv[5]
assert os.path.isfile(data_path), "File or folder not found"
rawData = sc.textFile(data_path).repartition(numPartitions=L)
t_rdd = rawData.map(lambda line: line.split(","))
inputPoints= t_rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()

print(f'{sys.argv[5]} D={D} M={M} K={K} L={L}')
n=inputPoints.count()
print(f'Number of points: {n}')
if n<=200000:
    listOfPoints=inputPoints.collect()
    exactOutliers(listOfPoints, D, M, K)
MRAApproxOutliers(inputPoints, D, M, K)
