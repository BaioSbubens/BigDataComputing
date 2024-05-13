from pyspark import SparkContext, SparkConf
import sys
import os
import random as random
import math
import time
import numpy as np

# SPARK SETUP
conf = SparkConf().setAppName('G050')
conf.set("spark.locality.wait", "0s")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

def distance(point1, point2):
    diff_i = point1[0] - point2[0]
    diff_j = point1[1] - point2[1]
    return diff_i * diff_i + diff_j * diff_j

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

def MRApproxOutliers(points, D, M):
    start = time.time()
    output_A = step_A(points,D)
    cells = output_A.collect()
    outliers, uncertain = step_B(cells,M)
    print(f'Number of sure outliers = {outliers}')
    print(f'Number of uncertain points = {uncertain}')
    finish = time.time()
    print(f'Running time of MRApproxOutliers = {((finish - start)*1000):.0f} ms')

def SequentialFFT(points, K):
    dis = np.zeros(len(points))
    C = [points[0]]
    for i in range(1, len(points)):
        dis[i] = distance(points[i], C[0])
    while len(C) < K:
        pos = np.argmax(dis)
        C.append(points[pos])
        for i in range(len(points)):
            dis[i] = min(dis[i], distance(points[i], points[pos])) 
    return C

def radius(inputPoints, C):
    centroids = C
    r = 0
    for el in inputPoints:
        dist = min(distance(el, center) for center in centroids)
        if dist > r:
            r = dist
    return [r]

def MRFFT(InputPoints,K):
    #Round1
    start_R1 = time.time()
    corset = InputPoints.mapPartitions(lambda x:SequentialFFT(list(x),K)).persist()
    finish_R1 = time.time() 
    #Round2
    cor1 = corset.collect()
    start_R2 = time.time()
    final_centroids = SequentialFFT(cor1,K)
    finish_R2 = time.time()
    #Round3
    start_R3 = time.time()
    C = sc.broadcast(final_centroids)
    rad_sqr = (InputPoints.mapPartitions(lambda x: radius(x, C.value))
              .reduce(lambda x,y: max(x,y)))
    rad = rad_sqr**0.5 # Since our distance function is not squared, for efficenty reason
    finish_R3 = time.time()
    print(f'Running time of MRFFT Round 1 = {((finish_R1 - start_R1)  *1000):.0f} ms')
    print(f'Running time of MRFFT Round 2 = {((finish_R2 - start_R2)  *1000):.0f} ms')
    print(f'Running time of MRFFT Round 3 = {((finish_R3 - start_R3)  *1000):.0f} ms')
    print(f'Radius = {round(rad,8)}')
    return rad

def main():

    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: python G050.py <file_name> <M> <K> <L> " 

    # SETTING GLOBAL VARIABLES
    
    M = int(sys.argv[2])
    K = int(sys.argv[3])

    # INPUT READING

    # 1. Read number of partitions
    L = int(sys.argv[4])
    
    # 2. Read input file and subdivide it into L random partitions
    data_path = sys.argv[1]
    #assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path).repartition(numPartitions=L).map(lambda line: line.split(","))
    inputPoints= rawData.map(lambda x: (float(x[0]),float(x[1]))).repartition(L).persist()
    
    print(f'{sys.argv[1]} M={M} K={K} L={L}')
    n = inputPoints.count()
    print(f'Number of points = {n}')
    D = MRFFT(inputPoints,K)
    MRApproxOutliers(inputPoints, D, M)
    
if __name__ == "__main__":
	main()