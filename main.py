import data_quotes_author as dq
import tf
import operator
import csv
from jellyfish import jaro_distance as jd

data, objects, sources, facst = dq.ReadData("all.csv")
truth_vector, tau_vector = tf.TruthFinder(data, len(sources), len(data))

d = {}
i = 0
for t in tau_vector:
    d[sources[i]] = t
    i = i+1

sorted_x = sorted(d.items(), key=operator.itemgetter(1))

est_truths = {}
for i in range(len(truth_vector)):
    est_truths[objects[i]] = facst[int(truth_vector[i])]

ground = []

f = open("res2.csv", "w")
for key, value in est_truths.items():
    print(key, type(value))
    print(value, type(value))
    f.write(key + "\t" + value + "\n")


with open("ground", 'r') as f:
    read = csv.reader(f, delimiter="\t")

    for row in read:
        ground.append([row[1], row[2]])

true = 0
false = 0
i = 0

for key, value in est_truths.items():
    if i%100 == 0:
        print(i)
    i += 1
    for t in ground:
        if jd(key, t[0]) > 0.98:
            if jd(value, t[1]) > 0.98:
                true += 1
            else:
                false += 1
            break

print("true:", true)
print("false:", false)
