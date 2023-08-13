with open('catalog212n.csv') as file:
    # Read space-delimited file and replace all empty spaces by commas
    data = file.read().replace(' ', ',')
    # Write the CSV data in the output file
    print(data, file=open('catalog115w.csv', 'w'))

