from PIL import Image
import numpy as np
import pickle
from heapq import nsmallest
import operator
import random
        
def convert_to_greyscale(original):
    greyscale = np.empty([original.shape[0], original.shape[1]], dtype='int')
    height = greyscale.shape[0]
    width = greyscale.shape[1]
    for x in range(height):
        for y in range(width):
            r, g, b = original[x, y]
            greyscale[x][y] = 0.21 * r + 0.72 * g + 0.07 * b
    
    return greyscale

def k_means(original, k):
    centroids = {}
    for i in range(k):
        centroids[i] = original[random.randint(0,original.shape[0])][random.randint(0,int(original.shape[1]/2))]
    consecutive = 0
    iter_num = 1
    count = 0
    while True:
        if iter_num == 10:
            count += 1
            print("Iteration", iter_num*count, ": Centroids are", centroids.values())
            iter_num = 0
        iter_num += 1
        isOptimal = True
        clusters = {}
        indices = {}
        for i in range(k):
            clusters[i] = []
            indices[i] = []
            
        for x in range(original.shape[0]):
            for y in range(int(original.shape[1]/2)):
                distances = [np.linalg.norm(original[x][y] - centroids[centroid]) for centroid in centroids]
                cluster_ind = distances.index(min(distances))
                clusters[cluster_ind].append(original[x][y])
                indices[cluster_ind].append((x,y))
                
        prev = dict(centroids)
        for cluster in clusters:
            if clusters[cluster] != []:
                centroids[cluster] = np.average(clusters[cluster], axis=0)
                
        for centroid in centroids:
            old = prev[centroid]
            curr = centroids[centroid]
            if np.sum(abs(curr - old)) > 0.0001:
                isOptimal = False
                consecutive = 0
                break
     
        if isOptimal:
            consecutive += 1
            if consecutive >= 10:
                break
            else:
                continue
    
    for centroid in centroids:
            centroids[centroid] = np.round(centroids[centroid], decimals=0).astype(np.uint8)
            
    print("Clustering done with centroids:", centroids.values())
        
    f = open("indices.pkl", 'wb')
    pickle.dump(indices, f)
    f.close()
    
    f = open("centroids.pkl", 'wb')
    pickle.dump(centroids, f)
    f.close()

def recolor(original, indices, centroids):
    recolored = original.copy()
    for centroid in centroids:
        for pixel in indices[centroid]:
            recolored[pixel[0]][pixel[1]] = centroids[centroid]
    
    return recolored[:,0:int(recolored.shape[1]/2)]

def predict(recolored, train, test):
    centers_and_neighbors = {}
    for x in range(1,train.shape[0]-1):
        for y in range(1, train.shape[1]-1):
            centers_and_neighbors[(x,y)] = np.array([train[x,y], train[x+1,y], train[x-1,y], train[x+1,y+1], train[x-1,y-1], train[x-1,y+1],train[x+1,y-1],train[x,y-1],train[x,y+1]])
    predicted = recolored.copy()
    predicted = predicted*0
    for x in range(1, test.shape[0]-1, 1):
        for y in range(1, test.shape[1]-1, 1):
            print(x,y)
            similarity = {}
            patch = np.array([test[x,y], test[x+1,y], test[x-1,y], test[x+1,y+1], test[x-1,y-1], test[x-1,y+1], test[x+1,y-1], test[x,y-1], test[x,y+1]])
            sample = dict(random.sample(centers_and_neighbors.items(), 1000))
            for center in sample:
                similarity[center] = np.linalg.norm(centers_and_neighbors[center] - patch)
            six_smallest = nsmallest(6, similarity, key=similarity.get)
            colors = {}
            for smallest in six_smallest:
                color = tuple(recolored[smallest[0],smallest[1]].tolist())
                colors[color] = 0
            for smallest in six_smallest:
                color = tuple(recolored[smallest[0],smallest[1]].tolist())
                colors[color] += 1
            
            max_value = max(colors.values())
            colorMax = [k for k,v in colors.items() if v == max_value]
            if len(colorMax) == 1:
                predicted[x,y] = np.asarray(colorMax[0])
            else:
                predicted[x,y] = np.asarray(tuple(recolored[six_smallest[0][0], six_smallest[0][1]].tolist()))
            
    return predicted

if __name__ == '__main__':
    original = np.array(Image.open('tiger.jpg'))
    greyscale = convert_to_greyscale(original)
    im = Image.fromarray(greyscale.astype(np.uint8))
    im.save('tiger_grey.jpg')
    indices = {}
    centroids = {}
    k_means(original, 5)
    with open(r'indices.pkl', 'rb') as input_file:
        indices = pickle.load(input_file)
    with open(r'centroids.pkl', 'rb') as input_file:
        centroids = pickle.load(input_file)
    print(centroids)
    recolored = recolor(original, indices, centroids)
    train = greyscale[:,0:int(greyscale.shape[1]/2)]
    test = greyscale[:,int(greyscale.shape[1]/2):]
    prediction = predict(recolored, train, test)
    print(np.unique(prediction[1], axis=0))
    result = np.concatenate((recolored, prediction), axis=1)
    im = Image.fromarray(result.astype(np.uint8))
    im.save('tiger_recolored.jpg')
    im.show()

    original_right = original[:, int(original.shape[1]/2):]
    colored = prediction[1:prediction.shape[0]-1, 1:prediction.shape[1]-1]
    original_right = original_right[1:original_right.shape[0]-1,1:original_right.shape[1]-1]
    err = np.zeros((colored.shape[0], colored.shape[1], 3), dtype='int64')
    for x in range(err.shape[0]):
        for y in range(err.shape[1]):
            err[x][y][0] = np.power(int(original_right[x][y][0]) - int(colored[x][y][0]),2)
            err[x][y][1] = np.power(int(original_right[x][y][1]) - int(colored[x][y][1]),2)
            err[x][y][2] = np.power(int(original_right[x][y][2]) - int(colored[x][y][2]),2)
    avg_err = np.sum(err)/(err.shape[0]*err.shape[1]*3)
    print(avg_err)
    
