import pickle as pkl
import csv

def yn_number(string):
    string = string.lower()
    if(string == 'yes'):
        return 1
    elif(string == 'no'):
        return -1
    else:
        raise Exception('Invalid boolean value: ' + string)
    

def region_number(string):
    if(string == 'northeast'):
        return 1
    elif(string == 'northwest'):
        return 2
    elif(string == 'southeast'):
        return 3
    elif(string == 'southwest'):
        return 4
    else:
        raise Exception("Invalid region: " + string)

def sex_number(string):
    if(string == 'male'):
        return 1
    if(string == 'female'):
        return -1
    else:
        raise Exception("Invalid gender: " + string)

#open insurance.csv and transform to feature vector representation
def read_and_transform():
    with open('insurance.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = [[int(row[0]), sex_number(row[1]),float(row[2]),int(row[3]), \
              yn_number(row[4]), region_number(row[5]),float(row[6])] \
             for row in reader] \
             
    return rows

obj = read_and_transform()
with open('insurance.p','wb') as file:
    pkl.dump(obj,file)
