from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random
import math
import numpy as np
from collections import Counter

# Global parameters
  # To be set via command line

# Sticky Sampling parameters
S = {}
current_bucket = 0
bucket_width = 0
epsilon = 0
phi = 0
delta = 0

# Reservoir Sampling parameters
reservoir = []
reservoir_size = 0

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    global streamLength, histogram, current_bucket, bucket_width, S, epsilon, phi, delta, reservoir, reservoir_size
    batch_items = batch.map(lambda s: int(s)).collect()

    # Update the histogram and reservoir
    for item in batch_items:
        if streamLength >= THRESHOLD:
            stopping_condition.set()
            return
        streamLength +=1
        histogram[item] = histogram.get(item, 0) + 1

        if len(reservoir) < reservoir_size:
            reservoir.append(item)
        else:
            rand_index = random.randint(0, streamLength - 1)
            if rand_index < reservoir_size:
                reservoir[rand_index] = item
        '''
        # Sticky Sampling
        if item in S:
            S[item] += 1
        else:
            if random.random() < phi:
                S[item] = 1

        if streamLength[0] % bucket_width == 0:
            current_bucket += 1
            for key in list(S.keys()):
                if S[key] < current_bucket:
                    del S[key]
'''

if __name__ == '__main__':
    assert len(sys.argv) == 6, "USAGE: n, phi, epsilon, delta, port"

    # Input parameters
    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])
    
    THRESHOLD = n
    reservoir_size = math.ceil(1 / phi)
    bucket_width = math.ceil(1 / epsilon)
    
    # Spark configuration
    conf = SparkConf().setMaster("local[*]").setAppName("GxxxHW3")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    # Stopping condition
    stopping_condition = threading.Event()
    
    # Data structures to maintain the state of the stream
    streamLength = 0  # Stream length (an array to be passed by reference)
    histogram = {}  # Hash Table for the distinct elements
    
    # Process the stream
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    # Start the streaming engine
    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, True)

    # Compute and print final statistics
    true_frequent_items = sorted([item for item in histogram if histogram[item] / streamLength >= phi])
    epsilon_frequent_items = sorted(S.keys())

    print("INPUT PROPERTIES")
    print("n =", streamLength, "phi =", phi, "epsilon =", epsilon, "delta =", delta, "port =", portExp)
    print("EXACT ALGORTIHM")
    print("Number of items in the data structure =", len(histogram))
    print("Number of true frequent items =", len(true_frequent_items))
    print("True frequent items:")
    for item in true_frequent_items:
        print(item)
    
    print("RESERVOIR SAMPLING")
    print("Size m of the sample =", reservoir_size)
    reservoir_dict = Counter(reservoir)
    print("Number of estimated frequent items =",len(reservoir_dict))
    print("Estimated frequent items:")
    for item in sorted(reservoir_dict):
        sign = "+" if item in true_frequent_items else '-'
        print(item, sign)
    
    print("STICKY SAMPLING")
    print("Number of items in the Hash Table =",)
    print("Number of estimated frequent items =",)
    print("Estimated frequent items:")
    for item in sorted():
        sign = "+" if item in true_frequent_items else '-'
        print(item, sign)
