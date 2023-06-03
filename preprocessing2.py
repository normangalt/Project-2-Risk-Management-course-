import folium
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import branca.colormap as cm
from sklearn.cluster import KMeans

# Read the CSV file
csv_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\combined_data.csv'
data = pd.read_csv(csv_file)

beijing_map = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
beijing_map2 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

beijing_map5 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
beijing_map6 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

beijing_map7 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
beijing_map8 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

beijing_map9 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
beijing_map10 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

beijing_map11 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
beijing_map12 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)


# Iterate through the rows and add markers to the map
for index, row in data.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']

    # Add a marker for each coordinate
    folium.Circle([longitude, latitude], color = "blue").add_to(beijing_map)
    folium.Circle([longitude, latitude], color = "blue").add_to(beijing_map2)
    folium.Circle([longitude, latitude], color = "blue").add_to(beijing_map5)
    folium.Circle([longitude, latitude], color = "blue").add_to(beijing_map6)
    folium.Circle([longitude, latitude], color = "blue").add_to(beijing_map9)
    folium.Circle([longitude, latitude], color = "blue").add_to(beijing_map8)

max_latitude = max(data["Latitude"])
min_latitude = min(data["Latitude"])

max_longitude = max(data["Longitude"])
min_longitude = min(data["Longitude"])

diff_longitude = max_longitude - min_longitude
diff_latitude = max_latitude - min_latitude

n_longitude = int(diff_longitude // 0.01) + 1
n_latitude = int(diff_latitude // 0.01) + 1

blocks = np.zeros((n_longitude, n_latitude))

def get_block(latitude, longitude):
    longitude_block = int((longitude - min_longitude) // 0.01)
    latitude_block = int((latitude - min_latitude) // 0.01)

    return longitude_block, latitude_block

for id in data['Id'].unique():
    taxi_data = data.loc[data["Id"] == id]
    taxi_row = taxi_data.iloc[0]

    taxi_data = taxi_data.iloc[1:]

    date = taxi_row["Timestamp"]
    date = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    lat = taxi_row["Latitude"]
    long = taxi_row["Longitude"]

    block_id_x, block_id_y = get_block(lat, long)
    block_id = block_id_x*n_latitude + block_id_y

    dwelling_time = 0

    for taxi_row in taxi_data.iterrows():
        taxi_row = taxi_row[1]
        date_new = taxi_row["Timestamp"]
        date_new = dt.datetime.strptime(date_new, '%Y-%m-%d %H:%M:%S')

        lat_new = taxi_row["Latitude"]
        long_new = taxi_row["Longitude"]

        block_id_x_new, block_id_y_new = get_block(lat_new, long_new)

        block_id_new = block_id_x_new*n_latitude + block_id_y_new
        time_difference = (date_new - date).total_seconds() / 60

        dwelling_time += time_difference
        if block_id == block_id_new:
            if dwelling_time >= 20:
                blocks[block_id_x_new, block_id_y_new] += 1
                dwelling_time = 0

        date = date_new
        lat = lat_new
        long = long_new

        block_id_x = block_id_x_new
        block_id_y = block_id_y_new

        block_id = block_id_new

np.savetxt("blocks.txt", blocks)

ranges_latitude = np.arange(min_latitude, max_latitude, 0.01)
ranges_longitude = np.arange(min_longitude, max_longitude, 0.01)

start_latitude = ranges_latitude[0]
start_longitude = ranges_longitude[0]

beijing_map3 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

folium.Circle([start_longitude, start_latitude], color = "red").add_to(beijing_map3)
folium.Circle([start_longitude, start_latitude], color = "red").add_to(beijing_map)
folium.Circle([start_longitude, start_latitude], color = "red").add_to(beijing_map5)
folium.Circle([start_longitude, start_latitude], color = "red").add_to(beijing_map8)
folium.Circle([start_longitude, start_latitude], color = "red").add_to(beijing_map7)
folium.Circle([start_longitude, start_latitude], color = "red").add_to(beijing_map10)
for latitude_index in range(len(ranges_latitude)):
    end_latitude = ranges_latitude[latitude_index]

    for longitude_index in range(len(ranges_longitude)):
        end_longitude = ranges_longitude[longitude_index]
        folium.Circle([end_longitude, end_latitude], color = "red").add_to(beijing_map3)
        folium.Circle([end_longitude, end_latitude], color = "red").add_to(beijing_map)
        folium.Circle([end_longitude, end_latitude], color = "red").add_to(beijing_map5)
        folium.Circle([end_longitude, end_latitude], color = "red").add_to(beijing_map7)
        folium.Circle([end_longitude, end_latitude], color = "red").add_to(beijing_map8)
        folium.Circle([end_longitude, end_latitude], color = "red").add_to(beijing_map10)

        start_longitude = end_longitude

    start_latitude = end_latitude

ranges_latitude = np.arange(min_latitude + 0.005, max_latitude + 0.005, 0.01)
ranges_longitude = np.arange(min_longitude + 0.005, max_longitude + 0.005, 0.01)

start_latitude = ranges_latitude[0]
start_longitude = ranges_longitude[0]

centers = np.zeros((blocks.shape[0], blocks.shape[1], 2))

beijing_map4 = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map4)
folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map2)
folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map)
folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map6)
folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map7)
folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map8)
folium.Circle([start_longitude, start_latitude], color = "green").add_to(beijing_map11)
for latitude_index in range(len(ranges_latitude)):
    end_latitude = ranges_latitude[latitude_index]

    for longitude_index in range(len(ranges_longitude)):
        end_longitude = ranges_longitude[longitude_index]

        centers[longitude_index][latitude_index][0] = end_longitude
        centers[longitude_index][latitude_index][1] = end_latitude

        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map4)
        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map2)
        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map)
        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map6)
        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map7)
        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map8)
        folium.Circle([end_longitude, end_latitude], color = "green").add_to(beijing_map11)


        start_longitude = end_longitude

    start_latitude = end_latitude

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map2.html'
beijing_map2.save(map_file)

print(blocks.shape, centers.shape)

print("\n")

dwelled_blocks_centers = centers[blocks > 0]
np.savetxt("dwelled_blocks_centers.txt", dwelled_blocks_centers)

members_number = len(dwelled_blocks_centers)

with open("members_number.txt", "a") as output_file:
    output_file.write(str(members_number) + "\n")

for center in dwelled_blocks_centers:
    long, lat = center

    folium.Circle([long, lat], color = "yellow").add_to(beijing_map4)
    folium.Circle([long, lat], color = "yellow").add_to(beijing_map3)
    folium.Circle([long, lat], color = "yellow").add_to(beijing_map)
    folium.Circle([long, lat], color = "yellow").add_to(beijing_map5)
    folium.Circle([long, lat], color = "yellow").add_to(beijing_map6)
    folium.Circle([long, lat], color = "yellow").add_to(beijing_map7)
    folium.Circle([long, lat], color = "yellow").add_to(beijing_map12)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map.html'
beijing_map.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map3.html'
beijing_map3.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map4.html'
beijing_map4.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map5.html'
beijing_map5.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map6.html'
beijing_map6.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map7.html'
beijing_map7.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map8.html'
beijing_map8.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map9.html'
beijing_map9.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map10.html'
beijing_map10.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map11.html'
beijing_map11.save(map_file)

map_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\main_maps\beijing_map12.html'
beijing_map12.save(map_file)

wssS = {}

for index in range(1, members_number + 1):
    colors = cm.LinearColormap(colors=["green", "blue", "yellow", "purple", "pink", "black", "red"], vmin = 1, vmax = index)

    map_index = folium.Map(location=[39.9042, 116.4074], zoom_start=12)
    kmeans = KMeans(n_clusters = index, n_init = "auto", random_state = 0).fit(dwelled_blocks_centers)

    wss = kmeans.inertia_
    wssS[index] = wss

    labels = kmeans.labels_

    centers = kmeans.cluster_centers_

    plt.title(f"KMeans algorithm with k={index}")
    plt.xlabel("longitude")
    plt.ylabel("latitude")

    for class_index in range(index):
        labels_truth = [val==class_index for val in labels]
        class_members = [dwelled_blocks_centers[index]
                         for index in range(members_number)
                          if labels_truth[index]]
        
        x = [class_member[0] for class_member in class_members]
        y = [class_member[1] for class_member in class_members]

        plt.scatter(x, y, marker = "^")
        for index_member in range(len(class_members)):
            folium.Circle([float(x[index_member]), float(y[index_member])], color = colors(class_index+1)).add_to(map_index)

    for center in centers:
        plt.plot(center[0], center[1], color = "red", marker ="*")
        folium.Marker([float(center[0]), float(center[1])], color = "red").add_to(map_index)

    map_index.add_child(colors)

    plt.savefig(f"figures\\clusters\\clustering_plot_{index}.png")
    #plt.show()
    plt.clf()

    map_file = f'C:\\Users\\Roman Kypybida\\Desktop\\RiskManagement\\Project 3\\supporting_maps\\map_{str(index)}.html'
    map_index.save(map_file)

for index in range(11):
    plt.clf()

    plt.title("Graph of the clusters algorithms inertia")
    plt.xlabel("k - number of clusters")
    plt.ylabel("inertia - degree of the models explanation")
    plt.xticks(list(wssS.keys())[0::index+1], fontsize=10, rotation=90)
    plt.yticks(list(wssS.values())[0::index+1], fontsize=10, rotation=90)
    plt.plot(wssS.keys(), wssS.values())
    plt.savefig(f"figures\\wss\\wss_plot_{index}.png")
    plt.show()

plt.clf()
