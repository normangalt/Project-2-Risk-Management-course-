import os
import csv

directory = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\taxi_log_2008_by_id'
output_file = r'C:\Users\Roman Kypybida\Desktop\RiskManagement\Project 3\combined_data.csv'

# Get all files with the .txt extension in the directory
files = [file for file in os.listdir(directory) if file.endswith('.txt')]

# Open the output CSV file in write mode
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(["Id","Timestamp","Latitude","Longitude"])

    # Iterate through each text file
    for file in files:
        file_path = os.path.join(directory, file)

        # Open the text file in read mode
        with open(file_path, 'r') as text_file:
            # Read the content of the text file
            lines = text_file.readlines()

            for line in lines:
                line_mod = line.strip().split(",")
                writer.writerow(line_mod)

print("Combined data saved to", output_file)
