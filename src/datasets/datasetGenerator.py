from __future__ import print_function # for python 3 printing

import sys
import csv
import collections
from collections import defaultdict
import random
import numpy as np
import re



def buildDataArray(infile):
    data = []
    varNames = []

    artists = ["beyonce-knowles", "50-cent", "eazy-e", "casey-veggies", "fetty-wap", "flatbush-zombies", "bas", "frank-ocean", "grandmaster-flash", "childish-gambino", "clipse", "big-l", "aloe-blacc", "eminem", "future", "flobots", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"]
    
    #artists = ["beyonce-knowles", "50-cent", "eazy-e", "asap-ferg", "fetty-wap", "danny-brown", "flo-rida", "frank-ocean", "action-bronson", "childish-gambino", "clipse", "b-o-b", "aloe-blacc", "eminem", "future", "akon", "david-banner", "2-chainz", "drake", "big-sean", "dr-dre", "earl-sweatshirt", "chance-the-rapper", "common", "asap-rocky"]

    #print("Num Artists: " + str(len(artists)))

    artistCountsMap = defaultdict(int)

    artistArrays = {k:[] for k in artists}

    #with open(infile, newline = '') as csvfile:  # no newline = '' in python 2
    with open(infile) as csvfile:
    #with open(infile, 'r') as textfile: 
        fileReader = csv.reader(csvfile, delimiter = ",") 
        #rowCount = 0
        #for row in reversed(list(csv.reader(textfile, delimiter = ","))):
        i = 0
        j = 0
        for row in fileReader:
            i += 1
            #if rowCount > 1000:
            #break
            #if row[3] in artists:

            # don't use songs with really short (or nonexistent) lyrics, of which there are some
            if len(row[5]) < 400:
                continue

            # CAP number of songs by any given artist at 200? 
            # no -- reduces training data too much -- we want more data, not less
            #if artistCountsMap[row[3]] >= 40:
            #    continue

            artist = row[3]
            artistCountsMap[artist] += 1


            # CAP number of songs by any given artist in dataset at 200 -- we had many for beyonce, was over-predicting beyonce
            # (but what about 50-cent? tons of 50-cent songs, but wasn't really over-predicting him)
            # 50 cent has more false negatives than false positives; beyonce has more false positives than false negatives

            j += 1
            string = row[3] + " " + (row[5].replace('\n', ' ')) #[:600] # + '\n'
            
            # TAKE OUT EVERYTING IN BETWEEN BRACKETS!!!

            string = re.sub("\[(.*?)\]", "", string)
            #string = re.sub("\\", "", string)
            #data.append(row)
            #j += 1

            # POTENTIALLY RESTORE THIS LINE
            #data.append(string)
            if artist not in artists:
                artistArrays[artist] = []
            artistArrays[row[3]].append(string)


    for artist in artistArrays:
        #print(artistArrays[artistArray])
        # SHUFFLE songs -- because they came ordered by date, which complicates train/test division
        random.shuffle(artistArrays[artist])

        #if len(artistArrays[artist]) > 200:
        #    artistArrays[artist] = (artistArrays[artist])[:200]

        data += artistArrays[artist]
        random.shuffle(data)
        random.shuffle(data)

    print("i: " + str(i))
    print("j: " + str(j))
    print("LENGTH " + str(len(data)))
    print(artistCountsMap)
    #sortedEminemSongs = sorted(eminemSongs)
    #print(sortedEminemSongs)

#do artist counts !

# beyonce-knowles
# 50-cent
# eazy-e
# casey-veggies
# fetty-wap
# flatbush-zombies (a group, but...)
# bas
# frank-ocean
# grandmaster-flash
# childish-gambino
# clipse (maybe... group) 
# big-l (naybe...)
# aloe-blacc
# eminem
# blackstreet
# bone-thugs-n-harmony (maybe... group)
# future
# flobots (maybe... group)
# david-banner (maybe...)

# SWAPPING OUT bas because he only has 20 (17 post strip) songs -- add flo-rida
# SWAPPING OUT flatbush zombies because the're a GROUP and only have 36 (15 post strip) songs -- adding ab-soul
# SWAPPNG OUT ab-soul because he only has 39 songs (post strip) -- adding Danny Brown
# SWAPPING OUT Grandmaster flash because he only has 28 (15 post strip) songs -- adding Action Bronson
# SWAPPING OUT flobots because they only have 30 songs -- adding Akon
# SWAPPING OUT Casey-Veggies bbecause he only has 26 songs after stripping -- adding asap-ferg
# SWAPPING OUT Big L because he only has 36 songs post-strip -- adding b-o-b
#maybe swap out Big L?

# flo-rida? akon?
# maybe add asap-ferg?
#b-o-b?
#busta-rhymes?
#chief-keef?
#danny-brown?


    #tup = (np.array(data), varNames)
    #return tup
    return np.array(data)


def parseData(infile, outfile):

    #data, varNames = buildDataArray(infile)
    #data = np.swapaxes(data.copy(), 0, 1)
    data = buildDataArray(infile)
    #print(varNames)
    #print(data)
    #print("numSongs: " + str(len(data)))
    # only print final 200 rows -- to get 2014 and 2015 songs
    # have "and" be the delimiter for artist -- only include first artist
    # dash in between artist's first and last name so that it's one 'word'
    # print first 300 characters of lyrics, or first 40 words]

    #artistCounts = {artist:0 for artist in artists} # used to reserve some lyrics for dev set

    print([row for row in data], file = open(outfile, "a"))




def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python datasetGenerator.py <infile>.csv <outfile>.txt")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    parseData(inputfilename, outputfilename)


if __name__ == '__main__':
    main()