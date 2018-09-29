import csv
import matplotlib.pyplot as plt
import numpy as np

#converts yes/no to 1/0
def yn_number(string):
    string = string.lower()
    if(string == 'yes'):
        return 1
    elif(string == 'no'):
        return 0
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
        return 2
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

cases = read_and_transform()
age = [row[0] for row in cases]
sex = [row[1] for row in cases]
bmi = [row[2] for row in cases]
children = [row[3] for row in cases]
smoker = [row[4] for row in cases]
region = [row[5] for row in cases]
charges = [row[6] for row in cases]

children_avgs = []
for i in range(0, 1+max(children)):
    m = np.mean(list(map(lambda row: row[6], \
                         filter(lambda row: row[3] == i,cases)))) \

    children_avgs.append(m)

all_smokers = list(map(lambda row: row[6],filter(lambda row: row[4] > 0, cases)))
all_nonsmokers = list(map(lambda row: row[6],filter(lambda row: row[4] == 0, cases)))
all_northeast = list(map(lambda row: row[6],filter(lambda row: row[5] == 1, cases)))
all_northwest = list(map(lambda row: row[6],filter(lambda row: row[5] == 2, cases)))
all_southeast = list(map(lambda row: row[6],filter(lambda row: row[5] == 3, cases)))
all_southwest = list(map(lambda row: row[6],filter(lambda row: row[5] == 4, cases)))
mean_smokers = np.mean(all_smokers)
mean_nonsmokers = np.mean(all_nonsmokers)
mean_northeast = np.mean(all_northeast)
mean_northwest = np.mean(all_northwest)
mean_southeast = np.mean(all_southeast)
mean_southwest = np.mean(all_southwest)

plt.figure(1)
plt.title("AGE")
plt.xlabel("Age (years)")
plt.ylabel("Price (USD)")
plt.scatter(age,charges,marker='o')
plt.show()

plt.figure(2)
plt.title("BMI")
plt.xlabel("Body Mass Index")
plt.ylabel("Price (USD)")
plt.scatter(bmi,charges,marker='o')
plt.show()


plt.figure(3)
plt.title("CHILDREN")
plt.xlabel("Number of children")
plt.ylabel("Average Price (USD)")
plt.bar(range(0,1+max(children)),children_avgs)
plt.show()

plt.figure(4)
plt.title("SMOKER")
plt.xlabel("Nonsmoker or smoker?")
plt.ylabel("Average Price (USD)")
plt.bar([0,1],[mean_nonsmokers,mean_smokers])
plt.show()

plt.figure(5)
plt.title("REGION")
plt.xlabel("Northeast, Northwest, Southeast, Southwest")
plt.ylabel("Average Price (USD)")
plt.bar(range(1,5),[mean_northeast,mean_northwest,mean_southeast,mean_southwest])
plt.show()
