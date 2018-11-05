import numpy as np
import math
import sys
import copy, random
import matplotlib.pyplot as plt
import cv2
np.random.seed(sum([ord(c) for c in 'kishandh@buffalo']))
def euclidian_distance(p1, p2, brg=False):
    if brg:
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
    else:
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def get_mean(arr):
    x, y, l =0, 0, len(arr)
    for i in arr:
        x+=i[0]
        y+=i[1]
    return (x/l, y/l)

def get_mean2(arr):
    x, y, z, l = 0, 0, 0, len(arr)
    for i in arr:
        x+=i[0]
        y+=i[1]
        z+=i[2]
    return (x/l, y/l, z/l)


def invertDictionary(d):
    myDict = {}
    for i in d:
        value = d.get(i)
        myDict.setdefault(value,[]).append(i)
    return myDict

def plot(cluster_ass, centers, itr, var):
    center_2_points=invertDictionary(cluster_ass)
    colors = ['r', 'g', 'b']
    for h in range(0,3):
        x_axis,y_axis=zip(*center_2_points[h])
        plt.scatter(x_axis, y_axis, c=colors[h], marker="^")
        for a,b in zip(x_axis, y_axis):
            plt.text(a, b, ' ('+str(a)+','+str(b)+')')
    x_c, y_c=zip(*centers)
    plt.scatter(x_c, y_c, c=colors, marker="o")
    for a,b in zip(x_c, y_c):
        plt.text(a, b, ' ('+str(format(a, '.2f'))+','+str(format(b, '.2f'))+')')
    plt.savefig('task3_'+'iter'+str(itr)+'_'+var+'.jpg')
    plt.clf()

def find_and_plot_kmeans(x, centers, k):
    cluster_ass={}
    for i in x:
        cluster_ass[i]=None
    i=0
    count=0
    while(True):
        if i==0:
            temp = copy.deepcopy(cluster_ass)
        cluster=None
        min=sys.maxsize
        for j in range(0, k):
            ed=euclidian_distance(x[i], centers[j])
            if min > ed:
                cluster=j
                min=ed
            cluster_ass[x[i]]=cluster
        i+=1
        if i == len(x):
            count+=1
            i=0
            if temp==cluster_ass:
                break
            else:
                plot(cluster_ass, centers, count, 'a')
                for n in range(0,k):
                    centers[n]=get_mean(invertDictionary(cluster_ass)[n])
                plot(cluster_ass, centers, count, 'b')


def find_and_plot_kmeans2(x, centers, k):
        cluster_ass={}
        for i in x:
            for j in i:
                cluster_ass[tuple(j)]=None
        i=0
        count=0
        while(True):
            h=0
            while(h<512):
                if i==0 and h==0:
                    temp = copy.deepcopy(cluster_ass)
                cluster=None
                min=sys.maxsize
                for j in range(0, k):
                    ed=euclidian_distance(x[i][h], centers[j], True)
                    if min > ed:
                        cluster=j
                        min=ed
                cluster_ass[tuple(x[i][h])]=cluster
                h+=1
            i+=1
            if i == 512 and h== 512:
                count+=1
                i=0
                h=0
                if temp==cluster_ass:
                    break
                else:
                    for n in range(0,k):
                        centers[n]=get_mean2(invertDictionary(cluster_ass)[n])
                    print(centers)
        return cluster_ass


centers=[(6.2, 3.2), (6.6, 3.7), (6.5, 3.0)]

x=[(5.9, 3.2), (4.6, 2.9), (6.2, 2.8), (4.7, 3.2), (5.5, 4.2), (5.0, 3.0),
   (4.9, 3.1), (6.7, 3.1), (5.1, 3.8), (6.0, 3.0)]

find_and_plot_kmeans(x, centers, 3)

k_all = [3, 5, 10, 20]
for k in k_all:
    org_baboon=cv2.imread('baboon.jpg')
    centers=[]
    for i in range(k):
        centers.append(org_baboon[np.random.randint(1, 512)][np.random.randint(1, 512)])

    ca=find_and_plot_kmeans2(org_baboon, centers, k)

    for i in range(512):
        for j in range(512):
            org_baboon[i][j]=np.asarray(centers[ca[tuple(org_baboon[i][j])]])
    cv2.imwrite('task3_baboon_'+str(k)+'.jpg',org_baboon)
