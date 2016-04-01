import numpy as np

def parse_matrix(stream):
    result = []
    line = stream.next().strip()
    while not line.endswith(']'):
        result.append(map(float,line.split()))
        line = stream.next().strip()
    result.append(map(float,line.split()[:-1]))
    return np.array(result,dtype=np.float32)
    
def parse(stream):
    for line in stream:
        line = line.strip()
        if line.endswith('['):
            name = line.strip().split()[0]
            yield name,parse_matrix(stream)

def print_ark(name,array):
    print name,"["
    for i,row in enumerate(array):
        print " ",
        for cell in row:
            print "%0.6f"%cell,
        if i == array.shape[0]-1:
            print "]"
        else:
            print
 
