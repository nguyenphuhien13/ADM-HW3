import re
import math
import pandas as pd
import numpy as np
import nltk
import heapq
import pickle
import datetime
from nltk.corpus import stopwords
from operator import itemgetter 

# Loading the dictionary
with open('dictionary.pkl', 'rb') as f:
    data = pickle.load(f)
# Loading the dictionary with term count
with open('newdictionary.pkl', 'rb') as f:
    newdata = pickle.load(f)
# Read the csv file
ff = pd.DataFrame(pd.read_csv('Airbnb_Texas_Rentals.csv'))
ff = ff.fillna('0')
ff = ff.drop(['Unnamed: 0'], axis=1)
# Insert the date_post_1 column based on the date of listing
ff['month']=[(x.split()[0]).lower() for x in ff['date_of_listing']]
ff['month_number']=np.where(ff['month']=='may',"-05-01",
                             np.where(ff['month']=='june',"-06-01",
                                      np.where(ff['month']=='july',"-07-01",
                                              np.where(ff['month']=="august","-08-01",
                                                      np.where(ff['month']=="september","-09-01",
                                                              np.where(ff['month']=="october","-10-01",
                                                                      np.where(ff['month']=="november","-11-01",
                                                                              np.where(ff['month']=="december","-12-01",
                                                                                      np.where(ff['month']=="january","-01-01",
                                                                                              np.where(ff['month']=="february","-02-01",
                                                                                                      np.where(ff['month']=="march","-03-01",
                                                                                                              np.where(ff['month']=="april","-04-01","-01-01"))))))))))))


ff['year']=[x.split()[1] for x in ff['date_of_listing']]
ff['date_post']=ff['year']+ff['month_number']
ff['date_post_1']=[pd.to_datetime(x) for x in ff['date_post']]
# calculate the room rate for each listing and merge it to the data frame
ff['rate_num']=[str(d).replace("$","") for d in ff['average_rate_per_night']]
ff=ff.fillna('0')
ff['rate_num_1']=[pd.to_numeric(x) if x!="nan" else 0 for x in ff['rate_num'] ]
ff_means=pd.DataFrame(ff.groupby(['city'])['rate_num_1'].mean())
ff_means.columns=['Average_in_this_city']
ff=ff.merge(ff_means, left_on='city', right_on='city', how='left')

# FUNCTIONS----FUNCTIONS----FUNCTIONS------------------------------
#input = [word1, word2, ...]
#output = {word1: [pos1, pos2], word2: [pos1, pos2], ...}
def index_one_file(termlist):
    fileIndex = {}
    words = list(set(termlist))
    word_list = [x for x in termlist]
    for i in range(len(word_list)):
        for item in words:
            if item == word_list[i]:
                fileIndex.setdefault(item, []).append(i)
    return fileIndex

#input = {filename: [word1, word2, ...], ...}
#ouput = {filename: {word: [pos1, pos2, ...]}, ...}
def make_indices(dictionary):
    total = {}
    for filename in dictionary.keys():
        new = dictionary[filename]
        total[filename] = index_one_file(new)
    return total

# Dict reversal
#input = {filename: {word: [pos1, pos2, ...], ... }}
#output = {word: {filename: [pos1, pos2]}, ...}, ...}
def fullIndex(regdex):
    total_index = {}
    for filename in regdex.keys():
        for word in regdex[filename].keys():
            if word in total_index.keys():
                if filename in total_index[word].keys():
                    total_index[word][filename].extend(regdex[filename][word][:])
                else:
                    total_index[word][filename] = regdex[filename][word]
            else:
                total_index[word] = {filename: regdex[filename][word]}
    return total_index


# Search Engine
# Preprocess the search
def preprocess(search):
    search = search.lower().split()
    stop_words = set(stopwords.words('english'))
    lemma = nltk.wordnet.WordNetLemmatizer()
    search_lst = []
    for x in search:
        if not x in stop_words:
            x = re.sub("[^a-zA-Z]+", "*", x)
            if "*" in x:
                y = x.split('*')
                y[0]=lemma.lemmatize(y[0])
                search_lst.append(y[0])
                if len(y)>1:
                    y[1]=lemma.lemmatize(y[1])
                    search_lst.append(y[1])
            else:
                x = lemma.lemmatize(x)
                search_lst.append(x)
    search_lst = (' '.join(search_lst))
    return search_lst

#Input for the search
def search_eng_input(phrase):
    phrase = phrase.lower().split()
    n = len(phrase)
    list1, list2, list3 = [], [], []
    for x in phrase:
        x = preprocess(x)
        list1.append(x)
    for x in list1:
        if x in data.keys():
            list2.append(set(data[x].keys()))
    b = list2[0]
    for i in range(0,len(list2)):
        b = (b & list2[i])
    for x in b:
        list3.append(int(re.sub("[^0-9]+", "", x))-1)
    return list3

# Executing the query and return the result for conjunctive search
def exec_query_s_1(search):
    pd.set_option('display.max_colwidth', -1)
    l = []
    df = pd.DataFrame()
    l = (search_eng_input(search))
    if len(l)>0:
        df = ff[['title','description','city', 'url']].loc[l]
    if df.empty == False:
    	df.set_index('title', inplace=True)
    return df

# TF-IDF
def tf(term_count, total_count):
    return term_count / total_count

def idf(doc_count, contain_count):
    return math.log(doc_count / contain_count)

def tf_idf(term_count, total_count, doc_count, contain_count):
    if total_count == 0: total_count = 1
    if contain_count == 0: contain_count = 1    
    return round(tf(term_count, total_count) * idf(doc_count, contain_count),2)


# return the number of words in a document when input in the name
def total_count(filename):
    total = 0
    inverse_data = fullIndex(data)		#inverse the data
    if filename in inverse_data.keys():
        value = inverse_data.get(filename, 0) #get the sub_dict
        for k, v in value.items():
            total += len(v) # count the number of term in a document
        return total
    else:
        return 0

# return the number of documents that contain a certain word when input in a term
def contain_count(term):
    if term in data.keys():
        return len(data[term].keys()) 
    else:
        return 0

# functions for returning the search with ranking similarity scores

#creating doc vectors
def doc_vec(query):
    lemma = nltk.wordnet.WordNetLemmatizer()
    querylist = query.split()
    query = search_eng_input_1(query) # return the list of documents matched first
    query = [x+1 for x in query] # +1 for the correct position
    doc = {}
    docvec = [0] * len(querylist)
    for index, word in enumerate(querylist):
        word = lemma.lemmatize(word)
        word = word.lower()
        try:
            subvec = []
            value = newdata[word]# get {doc1:tf-idf, doc2: tf-idf} of each word in search query
            for k, v in value.items():
                for i in query: # loop all the documents'ids that search gives
                    key = ('filtered_doc_%s'%str(i))    
                    if key ==  k:      # if the id is in the dict
                        subvec.append(v) # append the score to the vector = [[tf-idf1,tf-idf2,..],[tf-idf1,tf-idf2,..],..]
            subvec += [0] * (len(query) - len(subvec))  # make the vectors equal in length for not found instances           
            docvec[index] = subvec
            del subvec
        except KeyError:
        	docvec[index] = [0]*len(value.keys())	# if the word not in dict, create a zero vector
    # this loop return the dict with format {doc1:vector1,doc2:vector2,...} for the query
    for index in range(len(docvec[0])):
        sub_vec = [item[index] for item in docvec]
        doc.update({query[index]:sub_vec})

    return doc

#create query vector
def query_vec(query):
    pattern = re.compile('[\W_]+') # for faster search function
    query = pattern.sub(' ',query)
    querylist = query.split()
    b = len(querylist)
    c = 18259 #total number of documents
    queryvec = [0]*b
    for index,word in enumerate(querylist):
        a = querylist.count(word)
        d = contain_count(word)
        wordtfidf = tf_idf(a,b,c,d) # tf-idf score for each word
        queryvec[index] = wordtfidf   
    return queryvec

def dotproduct(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0
    return sum([x*y for x,y in zip(vec1, vec2)])

def magnitude(vec):
    return pow(sum(map(lambda x: x**2, vec)),.5)

# calculate the score of the results based on query
def generatescore(query):
    queryvec = query_vec(query)
    doc_vecs_dict = doc_vec(query)
    score_dict = {}
    for k, v in doc_vecs_dict.items():
        score = round(dotproduct(queryvec, v)/(magnitude(queryvec)*magnitude(v)),2)
        score_dict.update({k:score})
    return score_dict

# heap data structure to keep top k
def heappq(mysearch):
    query = search_eng_input(mysearch)
    k = 10 # default top k-element
    if k >= len(query):
        k = len(query)
    d = generatescore(mysearch)
    k_keys_sorted = heapq.nlargest(k, d.items(), key = itemgetter(1))
    key_lst, score_lst = [], []
    for i in range(k):
        key_lst.append(k_keys_sorted[i][0])
        score_lst.append(k_keys_sorted[i][1])
    return key_lst, score_lst

# executing tf_idf conjunctive search
def exec_tfidf_search(mysearch):
    key_lst, score_lst = heappq(mysearch)
    key_lst = [x-1 for x in key_lst] # to get the correct row in df
    pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame()
    if len(key_lst)>0:
        df = ff[['title','description','city', 'url']].loc[key_lst]
        df['similarity'] = score_lst
    if df.empty == False:
    	df.set_index('title', inplace=True)
    return df

#function for scoring for Step 4 of the task
def search_eng_input_1(search):
    search=search.lower().split()
    n=len(search)
    list1, list2, list3=[],[],[]
    
    for x in search:
        x=preprocess(x)
        list1.append(x)
    for x in list1:
        if x in data.keys():
            list2.append(set(data[x].keys()))
    if len(list2)>0:
        b=list2[0]
        for i in range(0,len(list2)):
            a=b
            b=(b & list2[i])
            if len(b)==0:
                b=a
                break
            else:
                a=b
        for x in b:
            list3.append(int(re.sub("[^0-9]+", "", x))-1)
    if len(list3)==0:
        list3=[1,2,3]
    return (list3)

# Executing search query
def exec_query_s_2(s):
    pd.set_option('display.max_colwidth', -1)
    l=[]
    df=pd.DataFrame()
    l=(search_eng_input_1(s))
    if len(l)>0:
        df=ff.loc[l]
    row_n=df.shape[0]
    list_idx, grades=[],[]
    for i in range(0,row_n):
        list_idx.append(i)
        grades.append(0)
    df.index=[list_idx]
    # to calculate how much time from posting date pass we need some day, data was submitted one yaer ago so that we can
    # choose some day in 2017
    df['Oper_date']=pd.to_datetime('2017-05-01')
    #https://livability.com/tx/real-estate/best-places-to-live-in-texas
    #in the website mentioned above there is top-10 of cities in texas
    list_of_best_cities=['Austin','Denton','Houston','Plano','Fort Worth','San Antonio','Dallas','Garland','College Station','Lubbock']
    for i in range(0,row_n):
        time_of_advert=(df['Oper_date'][i]-df['date_post_1'][i])
        #calculating how many days pass after posting, according to it grades are given
        if (time_of_advert <= '7 days').bool():
            grades[i]=5
        else:
            if (time_of_advert <= '31 days').bool():
                grades[i]=4
            else:
                if (time_of_advert <= '62 days').bool():
                    grades[i]=3
                else:
                    if (time_of_advert <='93 days').bool():
                        grades[i]=2
                    else:
                        if (time_of_advert<= '365 days').bool():
                            grades[i]=1
                        else:
                            grades[i]=0
        #comparing rate per night with average in this city           
        price_per_night=df['rate_num_1'][i]
        avg_price_in_city=df['Average_in_this_city'][i]
        if (price_per_night<=avg_price_in_city).bool():
            grades[i]=grades[i]+5
        else:
            if (price_per_night<=(avg_price_in_city+avg_price_in_city*0.2)).bool():
                grades[i]=grades[i]+3
            else:
                grades[i]=grades[i]-2
        #checking if city is in the top-10 of texas
        city=df['city'][i].values[0]
        if city in list_of_best_cities:
            grades[i]=grades[i]+10
        # if in the query there is meaning of renting full house or apartment we take into consideration number of bedrooms
        if ('house' in s)|('home' in s)|('apartment' in s):
            n_rooms=(df['bedrooms_count'][i].values[0])
            if n_rooms.isdigit()==True:
                grades[i]=grades[i]+int(n_rooms)
    df['GRADE']=grades
    #getting top-k
    key_lst, grades = heap_4(list_idx, grades)
    l_idx=[i for i in range(1,len(key_lst)+1)]
    df=df[['title','description','city', 'url','GRADE']].loc[key_lst].sort_values(by=['GRADE'],ascending=False)
    df=df.loc[key_lst].sort_values(by=['GRADE'],ascending=False)
    df['Ranking']=l_idx
    df=df.set_index('Ranking')
    df=df[['title','description','city', 'url']]
    return(df)

def heap_4(list_of_idx,list_of_grades):
    d={}
    k_keys_sorted=[]
    for i in range(0,len(list_of_idx)):
        d[list_of_idx[i]]=list_of_grades[i]
    k=10
    if k >= len(list_of_idx):
        k = len(list_of_idx)
    k_keys_sorted = heapq.nlargest(k, d.items(), key = itemgetter(1))
    key_lst, score_lst = [], []
    for i in range(k):
        key_lst.append(k_keys_sorted[i][0])
        score_lst.append(k_keys_sorted[i][1])
    return key_lst, score_lst