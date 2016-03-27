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


