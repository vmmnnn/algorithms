from math import sqrt

# find center of the cluster
def calculate_center(cluster):
    xs = []
    ys = []
    for cl in cluster:
        xs.append(cl[0])
        ys.append(cl[1])
    n = len(cluster)
    return [sum(xs) / n, sum(ys) / n]

# find centers for all k clusters
def upd_centers(clusters):
    centers = []
    for cl in clusters:
        cent = calculate_center(cl)
        centers.append(cent)
    return centers

def upd_clusters(k, centers, dataset):
    # create k clusters
    clusters = [[] for i in range (k)]

    # put next point into correct cluster according to its position
    for p in dataset:
        clusters[cl_num(p, centers)].append(p)

    return clusters

# find the nearest center
def cl_num(p, centers):
    min_l = l_count(p, centers[0])
    answ = 0
    for i in range(1, len(centers)):
        cur_l = l_count(p, centers[i])
        if cur_l < min_l:
            min_l = cur_l
            answ = i
    return answ

# calculate the distance
def l_count(p, cent):
    s = (p[0] - cent[0])**2 + (p[1] - cent[1])**2
    return sqrt(s)

def are_same(m1, m2):
    fl = True
    for i in range(len(m1)):
        if (m1[i][0] != m2[i][0]) or (m1[i][1] != m2[i][1]):
            fl = False
            break
    return fl

def print_res(clusters, centers):
    k = len(centers)
    print("    CENTERS: ")
    for i in range(k):
        print(i, ": ", centers[i])
    print("    CLUSTERS: ")
    for i in range(k):
        print(i, ": ", clusters[i])

f = open('data.txt', 'r')
dataset = []
for line in f:
    x, y = line.split(' ')
    dataset.append([float(x), float(y)])

# we need 3 clusters
k = 3

# take first k points as centers
centers = dataset[:k]


# algorithm starts
clusters = upd_clusters(k, centers, dataset)
same_fl = False

iters = 0
while not(same_fl):
    centers0 = centers.copy()
    centers = upd_centers(clusters)
    clusters = upd_clusters(k, centers, dataset)
    same_fl = are_same(centers0, centers)
    iters += 1

print("Iterations", iters)
print_res(clusters, centers)
