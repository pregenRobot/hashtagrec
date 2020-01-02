import csv
import numpy as np

with open('mined_data/queriedtags.csv', 'rt') as f:
    reader = csv.reader(f)
    queriedtags = list(reader)[0]
    
X = []
y = []
print(len(queriedtags))


for tag in queriedtags:
    with open("mined_data/"+tag+"_X.csv","rt") as f:
        reader = csv.reader(f)
        tag_vectors = list(reader)
        X.extend(tag_vectors)
    
    with open("mined_data/"+tag+"_y.csv","rt") as f:
        reader = csv.reader(f)
        tag_label = list(csv.reader(f))
        y.extend(tag_label)
        
    if(len(tag_label)<2):
        print(tag)
        print(len(tag_label))
        
print(len(y))
print(np.array(X).shape)
np.savetxt("mined_data/data_X.csv", np.array(X), delimiter=",", fmt='%s')
np.savetxt("mined_data/data_y.csv", np.array(y), delimiter=",", fmt='%s')

