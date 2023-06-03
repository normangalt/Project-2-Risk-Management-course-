import folium
import numpy as np
import matplotlib.pyplot as plt
import branca.colormap as cm
from sklearn.cluster import KMeans

members_number = None
with open("members_number.txt", "r") as output_file:
    members_number = int(output_file.readline().strip())

k = 15

dwelled_blocks_centers = np.loadtxt("dwelled_blocks_centers.txt")
blocks = np.loadtxt("blocks.txt")

blocks = blocks[blocks > 0]

colors = cm.LinearColormap(colors=["green", "blue", "yellow", "purple"], vmin = 1, vmax = k)

map_ = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
kmeans = KMeans(n_clusters = k, n_init = "auto", random_state = 0).fit(dwelled_blocks_centers)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

print(centers)
print("\n")
print(labels)

plt.title(f"KMeans algorithm with k={k}")
plt.xlabel("longitude")
plt.ylabel("latitude")

for class_index in range(k):
    labels_truth = [val==class_index for val in labels]
    class_members = [dwelled_blocks_centers[index]
                        for index in range(members_number)
                        if labels_truth[index]]

    blocks_class = blocks[labels_truth]
    np.savetxt(f"class_{class_index}_blocks.txt", blocks_class)

    x = [class_member[0] for class_member in class_members]
    y = [class_member[1] for class_member in class_members]

    plt.scatter(x, y, marker = "^")
    for index_member in range(len(class_members)):
        folium.Circle([float(x[index_member]), float(y[index_member])], color = colors(class_index+1)).add_to(map_)

for center in centers:
    plt.plot(center[0], center[1], color = "red", marker ="*")
    folium.Marker([float(center[0]), float(center[1])], color = "red").add_to(map_)

map_.add_child(colors)

plt.savefig(f"figures\\clusters\\clustering_plot.png")
#plt.show()
plt.clf()

map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\supporting_maps\\map.html'
map_.save(map_file)

np.savetxt("labels.txt", np.array(labels))
with open("supporting_numbers.txt", "w") as output_file:
    output_file.write(str(members_number) + '\n')
    output_file.write(str(k))

with open("wss_kmeans_division_1.txt", "w") as output_file:
    output_file.write(str(kmeans.inertia_) + '\n')
    output_file.write(str(kmeans.inertia_/k) + '\n')
