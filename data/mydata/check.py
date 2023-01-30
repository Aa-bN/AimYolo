import os

path1 = r'./images/val'
path2 = r'./labels/val'

p1 = os.listdir(path1)
p2 = os.listdir(path2)

t = 0

if len(p1) == len(p2):
    t = len(p1)
else:
    exit(-1)

counter = 0
for i in range(t):
    t1 = p1[i].split('.')[0]
    t2 = p2[i].split('.')[0]
    if t1 == t2:
        counter = counter + 1

print(counter)
