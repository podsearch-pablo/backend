import json
import math
overall = []
dct = {}

with open('./flaskr/vids.json') as info:
    dct = json.load(info)
    


# Initialize the overall data

for i in range(4,78):
    
    with open('./flaskr/json/' + str(i) + '.json') as json_file:
        data = json.load(json_file)
        data['info'] = dct[str(i)]
        overall.append(data)


def checkExists(word):
    """
    Prints the index of a given word in the corpus of data
    """
    print(data['text'].index(word))

def getTimes(word):
    """
    Returns the number of times a word appears
    """
    times = []
    i = 0
    ind = 0
    for jsonList in overall:

        for lst in jsonList['segments']:
            if (word in lst['text']):
                lst['info'] = jsonList['info']
                lst['dct'] = ind
                times.append(lst)
        ind+=1
    return times



def search(anses, ans, searchterm):
    """
    Sets up "anses", the TODO
    """
    for nd in range(len(ans)):

        timeStart = ans[nd]['start']

        nope = False
        
        
        current = ans[nd]['text']
        id = ans[nd]['id']
        newId = id
        count = 0
        while ('.' not in current):
            newId+=1
            if (count>10 or newId>= len(overall[ans[nd]['dct']]['segments'])):
                nope = True
                break

            current = current + overall[ans[nd]['dct']]['segments'][newId]['text']
            
            count+=1

        if (nope):
            continue
        
        if (current.index('.')>current.index(searchterm)):
            current = current[:current.index('.')+2]

        count = 0
        while (current.count('.') <2):
            id-=1
            if (count>10 or id<0):
                nope = True
                break
            current = overall[ans[nd]['dct']]['segments'][id]['text'] + current 
            timeStart = overall[ans[nd]['dct']]['segments'][id]['start']
            count+=1
        if (nope):
            continue
        youtubeURL = overall[ans[nd]['dct']]['info']['Link']

            
        anses.append((current.strip(), ans[nd]['dct'], timeStart, youtubeURL))



def listofids():
    """
    Prints a list of ids given the vids database
    """
    with open('./flaskr/vids.json') as info:
        dct = json.load(info)

    print(dct['0'].keys())
    lst = []
    for l in dct:
        lst.append(dct[l]['Link'][28:])
    print(lst)


#returns the list of everything
def getSearch(anses):
    """
    Returns the full list of all data given anses TODO
    """
    if (len(anses)==0):
        return []
    ind = anses[0][1]

    results = []
    vidResults = {}
    nextResult = {}
    vidResults['Name'] = overall[anses[0][1]]['info']['Name']
    vidResults['Clips'] = []





    for line in anses:
        if (line[1]!=ind):
            ind=line[1]
            
            results.append(vidResults)
            vidResults = {}
            vidResults['Name'] = overall[ind]['info']['Name']
            vidResults['Clips'] = []

        nextResult['text']=str(line[0])
        nextResult['link']='https://youtu.be/' + str(line[3][28:]) + '?t=' + str(int(line[2]))
        nextResult['exactTime']=line[2]
        vidResults['Clips'].append(nextResult)
        nextResult = {}
    
    results.append(vidResults)
    return results


from functools import cmp_to_key

def compare(dct1, dct2):
    """
    Compares the length of two dictionaries' clips (used for sorting)
    """
    if len(dct1['Clips']) < len(dct2['Clips']):
        return 1
    elif len(dct1['Clips']) > len(dct2['Clips']):
        return -1
    else:
        return 0


def masterSearch(searchterm):
    """
    The 'master search' for a term
    """

    ans = getTimes(searchterm)
    answers = []

    search(answers, ans, searchterm)
    unsortedResults = getSearch(answers)
    results = sorted(unsortedResults, key=cmp_to_key(compare))

    return results

if __name__ == '__main__':
    listofids()
    #masterSearch("NFT")