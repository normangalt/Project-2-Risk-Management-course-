import folium
import numpy as np
import sklearn.cluster as sk
import matplotlib.pyplot as plt
import branca.colormap as cm
from scipy.cluster.hierarchy import dendrogram

members_number = None
k = None
with open("supporting_numbers.txt", "r") as output_file:
    members_number = int(output_file.readline().strip())
    k = int(output_file.readline().strip())

dwelled_blocks_centers = np.loadtxt("dwelled_blocks_centers.txt")
labels = np.loadtxt("labels.txt")

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

class_labels = {}
class_division = {}
for class_index in range(k):
    plt.title(f"Hierarchycal clustering, class = {class_index}")
    labels_truth = [val==class_index for val in labels]
    class_members = [dwelled_blocks_centers[index_j]
                        for index_j in range(members_number)
                        if labels_truth[index_j]]

    class_labels[class_index] = labels_truth
    class_division[class_index] = class_members
    if len(class_members) == 1:
        continue

    hclust = sk.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(class_members)
    plot_dendrogram(hclust)

    plt.xlabel("width")
    plt.ylabel("height")
    #plt.show()
    plt.savefig(f"figures\\dendrograms\\dendrogram_class_{class_index}.png")
    plt.clf()

    print(hclust.n_clusters_)
    print(hclust.labels_)


main_map = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
main_map2 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

colors_list = ["green", "blue", "yellow", "purple", "red", "black"]
clusters_classes = {}
centers = []
colors_main_map = cm.LinearColormap(colors=colors_list, vmin = 1, vmax = 128)

cut_height = 0.05

with open("wss_kmeans_division_2.txt", "w") as output_file:
    total_inertia =  0
    for index in range(k):
        colors = cm.LinearColormap(colors=["green", "blue", "yellow", "red"], vmin = 1, vmax = index+1)
        map_index = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
        map_index2 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

        plt.title(f"Hierarchycal clustering, class = {index}")
        labels_truth = class_labels[index] 
        class_members = class_division[index]

        print(class_members)
        if len(class_members) == 1:
            clusters_classes[index] = [index]

            member = class_members[0]
            center = member

            centers.append(center)
            output_file.write(str(0) + "\n")
            print(1)
            print(index)

            plt.title(f"Hierarchy algorithm for class ={index}")
            plt.xlabel("longitude")
            plt.ylabel("latitude")

            plt.scatter(member[0], member[1], marker = "^", color = colors_list[0])   

            folium.Marker([float(center[0]), float(center[1])]).add_to(map_index2)
            folium.Marker([float(center[0]), float(center[1])]).add_to(main_map2)
    
            folium.Circle([float(class_members[0][0]), float(class_members[0][1])], color = colors_main_map((index+1)*4)).add_to(main_map)   
            folium.Circle([float(class_members[0][0]), float(class_members[0][1])], color = colors_main_map((index+1)*4)).add_to(main_map2)       

            plt.savefig(f"hclustering\\hclustering_figures\\hierarchy_algorithm_class_{index}.png")
            plt.show()

            map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\hclustering\\hclustering_maps\\hclustering_{index}.html'
            map_index.save(map_file)  

            map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\hclustering\\hclustering_maps\\hclustering_{index}_2.html'
            map_index2.save(map_file)  
            continue

        hclust = sk.AgglomerativeClustering(distance_threshold=cut_height, n_clusters=None).fit(class_members)

        labels = hclust.labels_
        clusters_n = hclust.n_clusters_

        clusters_classes[index] = labels

        print(clusters_n)
        print(labels)

        plt.title(f"Hierarchy algorithm for class ={index}")
        plt.xlabel("longitude")
        plt.ylabel("latitude")

        weights = np.loadtxt(f"class_{index}_blocks.txt")
        for class_index in range(clusters_n):
            labels_truth_ = [val==class_index for val in labels]
            class_members_ = [class_members[index_j]
                            for index_j in range(len(class_members))
                            if labels_truth_[index_j]]
            
            x = [class_member[0] for class_member in class_members_]
            y = [class_member[1] for class_member in class_members_]

            weights_class = weights[labels_truth_]
            weights_class = weights_class/np.sum(weights_class)
            kmeans = sk.KMeans(n_clusters = 1, n_init = "auto", random_state = 0).fit(class_members_, sample_weight = weights_class)

            center = list(kmeans.cluster_centers_)[0]
            centers.append(center)

            output_file.write(str(kmeans.inertia_) + "\n")

            total_inertia += kmeans.inertia_

            folium.Marker([float(center[0]), float(center[1])]).add_to(map_index2)
            folium.Marker([float(center[0]), float(center[1])]).add_to(main_map2)

            plt.scatter(x, y, marker = "^")
            for index_member in range(len(class_members_)):
                folium.Circle([float(x[index_member]), float(y[index_member])], color = colors(class_index+1)).add_to(map_index)
                folium.Circle([float(x[index_member]), float(y[index_member])], color = colors_main_map((index+1)*4+class_index+1)).add_to(main_map)
                
                folium.Circle([float(x[index_member]), float(y[index_member])], color = colors(class_index+1)).add_to(map_index2)
                folium.Circle([float(x[index_member]), float(y[index_member])], color = colors_main_map((index+1)*4+class_index+1)).add_to(main_map2)

        map_index.add_child(colors)

        plt.savefig(f"hclustering\\hclustering_figures\\hierarchy_algorithm_class_{index}.png")
        plt.show()

        map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\hclustering\\hclustering_maps\\hclustering_{index}.html'
        map_index.save(map_file)

        map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\hclustering\\hclustering_maps\\hclustering_{index}_2.html'
        map_index2.save(map_file)

    output_file.write(str(total_inertia) + "\n")

main_map.add_child(colors_main_map)

map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\hclustering\\hclustering_maps\\hclustering_main_map.html'
main_map.save(map_file)

map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\hclustering\\hclustering_maps\\hclustering_main_map_2.html'
main_map2.save(map_file)
