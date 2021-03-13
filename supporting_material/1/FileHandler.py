import csv


def read_from_file(file_name):
    data = []
    with open(file_name) as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        next(read, None)
        for row in read:
            data.append([float(row[0]),float(row[1])])
    return data
