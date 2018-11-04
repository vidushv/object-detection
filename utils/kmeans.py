import numpy as np
import os
import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt

'''
A collection of methods used for computing the IoU k-means clusters
'''
class YOLO_Kmeans:
    def __init__(self, cluster_number):
        self.cluster_number = cluster_number
    '''
    boxes   : dim=(n, 2) bounding box width and height
    clusters: dim=(k, 2) cluster width and height

    return  : dim(n, k)

    Returns the IoU score between all boxes and clusters
    '''
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))
        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    '''
    boxes   : dim=(n, 2) bounding box width and height
    clusters: dim=(k, 2) cluster width and height
    
    Returns the average IoU from each box to its cluster
    '''
    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    '''
    boxes: dim=(n, 2) bounding box width and height
    k    : the number of clusters
    
    Returns the k-means clusters using Lloyd's algorithm
    '''
    def kmeans(self, boxes, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, self.cluster_number))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, self.cluster_number, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(self.cluster_number):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

'''
Parses the directory dir_name to extract the bounding boxes, assuming
the DAC input format for all .xml files within
'''
def inport_boxes(dir_name):
    dimensions = []
    for path, directories, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".xml"):
                tree = ET.parse(path  + '/' + file)
                root = tree.getroot()
                obj = root.find("object")
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                xmax = float(bndbox.find("xmax").text)
                ymin = float(bndbox.find("ymin").text)
                ymax = float(bndbox.find("ymax").text)

                width = xmax - xmin
                height =  ymax - ymin
                dimensions.append([width, height])
    return np.array(dimensions)

if __name__ == "__main__":
    cluster_number = 9
    dir_name = "." ## directory that contains bounding box data

    all_boxes = inport_boxes(dir_name)
    kmeans = YOLO_Kmeans(cluster_number)
    clusters = kmeans.kmeans(all_boxes)
    clusters = clusters[np.lexsort(clusters.T[0, None])]

    kmeans.result2txt(clusters)

    print("K anchors:\n {}".format(clusters))
    print("Accuracy: {:.2f}%".format(
        kmeans.avg_iou(all_boxes, clusters) * 100))

    # UNCOMMENT FOR A PRETTY PLOT
    # plt.scatter(all_boxes[:, 0], all_boxes[:, 1], s=3, label='Bounding Boxes')
    # plt.scatter(clusters[:, 0], clusters[:, 1], c='green', s=30, label='Clusters')

    # plt.title('Bounding Box Clustering')
    # plt.ylabel('Height')
    # plt.xlabel('Width')
    # plt.legend(loc='upper left')
    # plt.show()