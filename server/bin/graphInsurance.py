
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

with open('insurance.p','rb') as file:
    cases = pkl.load(file)
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
