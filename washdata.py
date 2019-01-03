import numpy as np
import csv
import os

# specify the path
dataset_path = './data'

# training path
train_data_file = os.path.join(dataset_path, 'zip_train.csv')

# test path
test_data_file = os.path.join(dataset_path, 'zip_test.csv')

# outputpath
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# image's parameter
img_rows, img_cols = 16, 16

i = 0
header = ['label']
for i in range(256):
    header.append('image')

def load_train_dataset(data_file):
    with open(data_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        with open('output_train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            #if we don't open them at the beginning,we open it in the for loop
            #then,each time we write the line,latest line would overlap the former ones
            #we can only get the last row
            wr.writerow(header)
            for row in csv_reader:    #for each line
            #each row is a row,and it's a list which has only one element
                line=row[0].split(' ')
                del line[-1]    #the last one is a space
                wr.writerow(line)
                line.clear()
        myfile.close()
    csvfile.close()

def load_test_dataset(data_file):
    with open(data_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        with open('output_test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            #if we don't open them at the beginning,we open it in the for loop
            #then,each time we write the line,latest line would overlap the former ones
            #we can only get the last row
            wr.writerow(header)
            for row in csv_reader:    #for each line
            #each row is a row,and it's a list which has only one element
                line=row[0].split(' ')
                wr.writerow(line)
                line.clear()
        myfile.close()
    csvfile.close()

load_train_dataset(train_data_file)
load_test_dataset(test_data_file)