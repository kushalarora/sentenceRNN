import sys
import numpy as np
import random

def disform_sentences(input_filename, p1=80, p2=10, p3=10):
    infile = open(input_filename, 'r')
    output_filename = input_filename + ".deformed-%.1f-%.1f-%.1f" % (p1, p2, p3)
    outfile = open(output_filename, 'w')
    wordToIdx = {}
    idx = 0;
    input = []
    for line in infile:
        input.append(line)
        words = line.split()
        for word in words:
            if word not in wordToIdx:
                wordToIdx[word] = idx
                idx += 1

    V = wordToIdx.keys()
    VSize = len(V)

    pn = float(p1)/100
    ps = float(p2)/100
    pt = float(p3)/100
    for line in input:
        line = line.strip()
        words = line.split()
        sz = len(words)
        draw = np.random.choice([0,1,2], sz, p=[pn, ps, pt])
        for i, dr in enumerate(draw):
            if dr == 0:
                continue
            elif dr == 1:
                vIdx = random.randint(0, VSize-1)
                sub = V[vIdx]
                words[i] = sub

            elif dr == 2:
                vIdx = random.randint(0, sz - 1)
                sub = words[vIdx]
                words[vIdx] = words[i]
                words[i] = sub

        newline = " ".join(words)
#        print "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#        print line
#        print "======================================="
#        print newline
#        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        outfile.write(newline + "\n")
    return output_filename


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 5:
        disform_sentences(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif argc == 4:
        disform_sentences(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[3]))
    else:
        disform_sentences(sys.argv[1])
