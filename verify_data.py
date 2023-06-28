import csv

# Path to the CSV file
csv_file = './data/ESD.csv'

# Initialize a list to store the element counts
element_counts = []

# Open the CSV file and read the rows
with open(csv_file, 'r') as file:
    reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in reader:
        # Count the number of elements in the row
        element_count = len(row)
        
        # Store the element count in the list
        element_counts.append(element_count)

# Print the element counts
for i, count in enumerate(element_counts, 1):
    print(f"Row {i}: {count} elements")
