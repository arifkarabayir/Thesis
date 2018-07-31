import numpy as np

def ReadData(filename):
    fileHandle = open(filename)
    f = fileHandle.readline()
    objects = []
    data = []
    sources = []
    facts = []

    while f != '':
        dataline = f.strip()
        item = dataline.split("\t")
        obj = item[1]
        fact = item[2]
        source = item[3]

        if source not in sources:
            sources.append(source)

        if fact not in facts:
            facts.append(fact)

        if obj in objects:
            index = objects.index(obj)
            data[index].append([sources.index(source), facts.index(fact)])
        else:
            objects.append(obj)
            index = objects.index(obj)
            data.append([])
            data[index].append([sources.index(source), facts.index(fact)])

        f = fileHandle.readline()

    data=np.array([np.array(x) for x in data])
    

    fileHandle.close()
    data = np.asarray(data)
    return data, objects, sources, facts


if __name__ == "__main__":
    data, objects, sources, facts = ReadData("quotes.txt")
    for i in data:
        print(i)
    print("--------")
    print(sources)
    print("--------")
    print(objects)