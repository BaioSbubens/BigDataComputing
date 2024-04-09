from pyspark import SparkContext, SparkConf
import sys
import os
import random as random
import math
import time

def distance(point1, point2):
    diff_i = point1[0] - point2[0]
    diff_j = point1[1] - point2[1]
    return diff_i * diff_i + diff_j * diff_j

def ExactOutliers(points, D, M, K):
    start = time.time()
    l = []
    num_outliers = 0
    for i in points:
        s = 0
        for j in points:
            d = distance(i,j)
            if d < D**2:
                s += 1
            if s > M:
                break
        if s <= M:
            num_outliers += 1
            l.append(i)
    print("Number of outliers:", num_outliers)
    for k in sorted(l)[:K]:
        print(f'Point: {k}')
    finish = time.time()
    print(f'Running time of ExactOutliers = {((finish - start)*1000):.0f} ms')


def belong_cell(point,side):

    i=math.floor((point[0]/side))
    j=math.floor((point[1]/side))
    return [(i,j)]


def points_count_per_cell(cell):
    pairs_dict = {}
    for p in cell:
        if p not in pairs_dict.keys():
            pairs_dict[p] = 1
        else:
            pairs_dict[p] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]



def step_A(input,D):

    side = D/(2*math.sqrt(2))
    points_count = (input.flatMap(lambda x: belong_cell(x,side))
                    .mapPartitions(points_count_per_cell)
                    .groupByKey()
                    .mapValues(lambda vals: sum(vals)))        
    return points_count

def step_B(cells,M):
    outliers = 0
    uncertain = 0
    for center in cells:
        if center[1] <= M:
            i,j = center[0]
            tot_3x3 = 0
            tot_7x7 = 0
            for cell in cells:
                x,y = cell[0]
                if abs(x - i) <= 1 and abs(y - j) <= 1:
                    tot_3x3 += cell[1]
                elif abs(x - i) <= 3 and abs(y - j) <= 3:
                    tot_7x7 += cell[1]
                if tot_3x3 > M:
                    break
            tot_7x7 += tot_3x3
            if tot_7x7 <= M:
                outliers += center[1]
            elif tot_3x3 <= M:
                uncertain += center[1]
    return (outliers, uncertain)

def first_k_cells(output_A,K):
    ordered = (output_A.map(lambda x : (x[1], x[0]))
               .sortByKey()
               .take(K))
    return ordered

def MRApproxOutliers(points, D, M, K):
    start = time.time()
    output_A = step_A(points,D)
    cells = output_A.collect()
    outliers, uncertain = step_B(cells,M)
    print(f'Number of sure outliers = {outliers}')
    print(f'Number of uncertain outliers =  {uncertain}')

    for el in first_k_cells(output_A,K):
        print(f'Cell: {el[1]} Size = {el[0]}')

    finish = time.time()
    
    print(f'Running time of MRApproxOutliers = {((finish - start)*1000):.0f} ms')

def main():

    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 6, "Usage: python G050.py <D> <M> <K> <L> <file_name>" 

    # SPARK SETUP
    conf = SparkConf().setAppName('G050')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # SETTING GLOBAL VARIABLES
    D = float(sys.argv[1])
    M = int(sys.argv[2])
    K = int(sys.argv[3])

    # INPUT READING

    # 1. Read number of partitions
    L = sys.argv[4]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # 2. Read input file and subdivide it into L random partitions
    data_path = sys.argv[5]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path).repartition(numPartitions=L).map(lambda line: line.split(","))
    inputPoints= rawData.map(lambda x: (float(x[0]), float(x[1]))).cache()

    print(f'{sys.argv[5]} D={D} M={M} K={K} L={L}')
    n = inputPoints.count()
    print(f'Number of points = {n}')
    if n<=200000:
        listOfPoints = inputPoints.collect()
        ExactOutliers(listOfPoints, D, M, K)
    MRApproxOutliers(inputPoints, D, M, K)

if __name__ == "__main__":
	main()