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
    print(data['text'].index(word))

#get the times
def getTimes(word):
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



# Sets the 'anses' list up
def search(anses, ans, searchterm):


    for nd in range(len(ans)):

        timeStart = ans[nd]['start']
        print("TIME " + str(timeStart))

        nope = False
        
        
        current = ans[nd]['text']
        id = ans[nd]['id']
        newId = id

        
        count = 0
        while ('.' not in current and '?' not in current):
            newId+=1
            if (count>10 or newId>= len(overall[ans[nd]['dct']]['segments'])):
                print(str(ans[nd]['dct']))
                nope = True
                break

            current = current + overall[ans[nd]['dct']]['segments'][newId]['text']
            
            count+=1

        if (nope):
            continue

        if (current.index('?')!=-1):
            if (current.index('?')>current.index(searchterm)):
                current = current[:current.index('.')+2]
        elif (current.index('.')>current.index(searchterm)):
            current = current[:current.index('.')+2]

        count = 0
        while (current.count('.') + current.count('?') <2):
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
    with open('./flaskr/vids.json') as info:
        dct = json.load(info)

    print(dct['0'].keys())
    lst = []
    for l in dct:
        lst.append(dct[l]['Link'][28:])
    print(lst)


#returns the list of everything
def getSearch(anses):
    if (len(anses)==0):
        return []
    ind = anses[0][1]

    results = []

    vidResults = {}



    nextResult = {}

    vidResults['Name'] = overall[anses[0][1]]['info']['Name']
    vidResults['Clips'] = []

    print(vidResults['Name'])




    for line in anses:
        if (line[1]!=ind):
            print()
            print()
            print()
            print()
            ind=line[1]
            
            results.append(vidResults)
            vidResults = {}
            vidResults['Name'] = overall[ind]['info']['Name']
            vidResults['Clips'] = []
            
            print(overall[ind]['info']['Name'])
            
            print()
        
        nextResult['text']=str(line[0])
        nextResult['link']='https://youtu.be/' + str(line[3][28:]) + '?t=' + str(int(line[2]))
        nextResult['exactTime']=line[2]
        
        vidResults['Clips'].append(nextResult)
        
        nextResult = {}
        
        print(str(line[0]))
        print('https://youtu.be/' + str(line[3][28:]) + '?t=' + str(int(line[2])))
        print(line[2])
        print()

    results.append(vidResults)
    return results


from functools import cmp_to_key

#custom searching feature 
def compare(dct1, dct2):
    
    if len(dct1['Clips']) < len(dct2['Clips']):
        return 1
    elif len(dct1['Clips']) > len(dct2['Clips']):
        return -1
    else:
        return 0


def masterSearch(searchterm):
    print("AT MAIN")


    searchingFor = searchterm
    ans = getTimes(searchingFor)

    anses = []

    search(anses, ans,searchingFor)


    results = getSearch(anses)



    results = sorted(results, key=cmp_to_key(compare))

    for val in results:
        print("NAME IS " + str(val['Name']))
        for adct in val['Clips']:
            print(adct['text'])
            print(adct['link'])
        print()
        print()
    return results

if __name__ == '__main__':
    listofids()
    #masterSearch("NFT")