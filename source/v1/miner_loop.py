import pandas as pd 
import csv 
import subprocess
import os

hashtaglist = pd.read_csv("/Users/tamimazmain/Projects/datasets/top_twitter_hashtags/Top_hashtag.csv")

hashtags = hashtaglist.sort_values("Posts",ascending=False)["Hashtag"][:100]

with open('mined_data/queriedtags.csv', 'rt') as f:
    reader = csv.reader(f)
    queriedtags = list(reader)[0]
    
print("Already queried: ")
print(queriedtags)

hashtags = [hashtag for hashtag in hashtags if not set([hashtag]).issubset(queriedtags)]
for i,hashtag in enumerate(hashtags):
    stream = os.popen("python miner.py "+hashtag)
    output = stream.read()
    print(output)
    
    streamw = os.popen("echo \","+hashtag+"\" >> mined_data/queriedtags.csv")
    # streamw = os.popen("sed \"s/$/"+hashtag+"/\" mined_data/queriedtags.csv > mined_data/queriedtags.csv")
    print(streamw)
    print("FINISHED " + str(i+1) + "/" + str(len(hashtags)))
    



# ##Remove later
# alreadyqueried = set(["beach","family","friend","love","yellow"])
# ##

