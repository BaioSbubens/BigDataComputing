from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random
import math
import numpy as np
from collections import Counter

def process_batch(batch):
    global streamLength, histogram, reservoir, sticky
    batch_items = batch.collect()

    for item in batch_items:
        if streamLength >= n:
            stopping_condition.set()
            return
        streamLength +=1

        # True frequent items
        histogram[item] = histogram.get(item, 0) + 1

        # Reservoir sampling
        if len(reservoir) < reservoir_size:
            reservoir.append(int(item))
        else:
            rand_index = random.randint(0, streamLength - 1)
            if rand_index < reservoir_size:
                reservoir[rand_index] = int(item)
        
        # Sticky Sampling
        r = (math.log(1/(delta * phi)))/epsilon
        if item in sticky:
            sticky[item] += 1
        else:
            if random.random() <= r/n:
                sticky[item] = 1
    


if __name__ == '__main__':
    assert len(sys.argv) == 6, "USAGE: n, phi, epsilon, delta, port"

    # Input parameters
    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])
    
    # Spark configuration
    conf = SparkConf().setMaster("local[*]").setAppName("G050HW3")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    # Stopping condition
    stopping_condition = threading.Event()

    # Data structures to maintain the state of the stream
    streamLength = 0  # Stream length

    # True frequent items parameters
    histogram = {} 

    # Reservoir Sampling parameters
    reservoir = []
    reservoir_size = math.ceil(1 / phi)

    # Sticky Sampling parameters
    sticky = {}

    # Process the stream
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda batch: process_batch(batch))
    
    # Start the streaming engine
    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)

    # Compute and print final statistics
    
    print("INPUT PROPERTIES")
    print("n =", streamLength, "phi =", phi, "epsilon =", epsilon, "delta =", delta, "port =", portExp)
    print("EXACT ALGORTIHM")
    print("Number of items in the data structure =", len(histogram))
    true_frequent_items = sorted([int(item) for item in histogram if histogram[item] / streamLength >= phi])
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
    print("Number of items in the Hash Table =", len(sticky))
    s_frequent = [int(item) for item in sticky if sticky[item] >= (phi - epsilon)*n]
    print("Number of estimated frequent items =", len(s_frequent))   
    print("Estimated frequent items:")
    for item in sorted(s_frequent):
        sign = "+" if item in true_frequent_items else '-'
        print(item, sign)
