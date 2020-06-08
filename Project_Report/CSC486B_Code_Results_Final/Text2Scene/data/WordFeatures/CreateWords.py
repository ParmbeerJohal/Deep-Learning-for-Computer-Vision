#Created by Sean Mcfarlane

import string

if __name__ == "__main__":
    counts = dict()
    with open('../Sentences_1002.txt', 'r') as f:
        for line in f:
            lowLine = line.lower()
            words = lowLine.split()
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in words]
            for word in stripped:
                if(word in counts):
                    counts[word] += 1
                else:
                    counts[word] = 1

    ## Remove entries with less than 5 occurrances.
    remove = [k for k in counts if counts[k] < 5]
    for k in remove: del counts[k]

    sortedDict = sorted(counts, key=counts.get, reverse=True)
    with open('Words.txt', 'w') as file:
        for word in sortedDict:
            file.write(str(counts[word])+"\t"+str(word)+"\n")

    ##
    ##  Creating word buckets
    ##
    with open('../Sentences_1002.txt', 'r') as input:
        with open('WordFeatures_'+str(len(counts))+'.txt', 'w') as output:
            for line in input:
                lowLine = line.lower()
                words = lowLine.split()
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in words]
                output_string = ""
                for word in counts:
                    if word in stripped:
                        output_string = output_string+"1\t"
                    else:
                        output_string = output_string+"0\t"
                output.write(output_string+"\n")
