
# coding: utf-8

# In[100]:

import streamlit as st
import re
import numpy as np
import pandas as pd
from stemmer import Stemmer
import warnings
warnings.filterwarnings("ignore")
data = pd.read_excel('Gujarati_Dimensionality_Reduction.xlsx')

st.header("""Medical Chatbot in Gujarati""")
st.title("રોગ શોધક ચેટબોટ ") 

input_text =  st.text_input("Enter Disease or symptoms", value= "રોગ") 

s1="રોગ"
if input_text.startswith(s1): 
    flag=1
    stemmer = Stemmer()
    stemmed_text = stemmer.stem(input_text)
    stemmed_words = re.split(r'[;|,|\s]\s*', stemmed_text) 

    diseases_dire=[]

    for i in range(len(stemmed_words)): 
        for col in data['રોગ']: 
            if(stemmed_words[i]==col and stemmed_words[i]!='રોગ'):
                diseases_dire.append(stemmed_words[i])

    st.write("રોગો મેળ ખાતા ફોર્મ ડેટાસેટ : ",diseases_dire)
    diseases_dire=np.array(diseases_dire)

    data2 = pd.read_excel ("Gujarati_Dataset2.xlsx")

    Y = data2[data2.columns[0]].to_numpy()
    disease = Y.tolist()

    lendisease2=len(disease)

    columnnum=[]
    diseases_dire=np.array(diseases_dire)
    for i in range(lendisease2):
        for j in diseases_dire:
            if j ==  disease[i]:
                columnnum.append(i)

    disease_row=data2.loc[columnnum,:]
    disease_row=disease_row.to_numpy()
    for i in range(len(columnnum)):
        
        st.write("\nરોગ : \n")
        st.write(disease_row[i][0])
        st.write("\nવર્ણન : \n")
        st.write(disease_row[i][1])
        st.write("\nલક્ષણો : \n")
        st.write(disease_row[i][2])
        st.write("\nઘરેલું ઉપાય : \n")
        st.write(disease_row[i][3])
        st.write("\nસારવાર : \n")
        st.write(disease_row[i][4])

else:       
    flag=0
    stemmer = Stemmer()
    input_text=input_text.replace("અને", ",")
    stemmed_text = stemmer.stem(input_text)
    stemmed_words = re.split(r'(;|,|\s)\s*', stemmed_text) 
    st.write()
    #st.write("સ્ટેમ પછી : ",stemmed_text)
   
    f = open("guj_pos_tag.txt",mode='r',encoding='UTF-8')
    data = f.read()
    sentences = data.split('\n')[1:]
    words = []
    for s in sentences:
        words.append(s.split('\t')[1])

    import re
    for i in range(len(sentences)):
        words[i] = re.sub(r'[^.A-Zઁ-૱\\,_-]',' ',words[i])
    pairs = []
    for i in range(len(words)):
        pairs.append(words[i].split(" "))
    tagged_guj_sentences = []

    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            if len(pairs[i][j].split("\\")) ==2:
                k,v = pairs[i][j].split("\\")
                tagged_guj_sentences.append((k,v))

    vocab=[word for word,tag in tagged_guj_sentences]
    tags=[tag for word,tag in tagged_guj_sentences]

    from stemmer import Stemmer
    stemmer = Stemmer()
    stem_words = []
    for v in vocab:
        stem_words.append(stemmer.stem(v))
    noun_words_intxt = dict(zip(stem_words,tags))
    noun_words_intxt = [(k, v) for k, v in noun_words_intxt.items()]  #convert to list
    #st.write("પોસ્ટગર : ",noun_words_intxt)

    from difflib import SequenceMatcher
    compared_words=[]
    maxn=0
    for j in range(len(stemmed_words)):
        for i in range(len(noun_words_intxt)):
            if(noun_words_intxt[i][0]!='મ'):
                if(noun_words_intxt[i][1]=='N_NN' or noun_words_intxt[i][1]=='RD_PUNC' ):
                    #st.write(noun_words_intxt[i][0],stemmed_words[j])
                    if(stemmed_words[j]==noun_words_intxt[i][0]):
                        #st.write(noun_words_intxt[i][0],stemmed_words[j])
                        compared_words.append(stemmed_words[j])
    #st.write()
    #st.write("ફક્ત નોઉન્સ : ",compared_words)
    #st.write()

    def listToString(s):  

        str1 = ""  

        for ele in s:  
            str1 =str1+ele+" "   

        return str1  

    listnoun=listToString(compared_words)
    symptoms_byuser =  re.split(r'[,]\s*', listnoun) 
    symptoms_byuser=np.array(symptoms_byuser)
    for i in range(len(symptoms_byuser)):
        symptoms_byuser[i]=symptoms_byuser[i].strip()
        
    symptoms_byuser=symptoms_byuser.tolist()
    #st.write("વપરાશકર્તાઓ દ્વારા લક્ષણો : ",symptoms_byuser)
    #st.write()

    symptoms_byuser=np.array(symptoms_byuser)

    from difflib import SequenceMatcher
    
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    symptoms=[]
    max_n=[]
    symptom=0

    for i in range(len(symptoms_byuser)): 
        maxn=0
        for col in data.columns: 
            #st.write(symptoms_byuser[i])
            ratio = SequenceMatcher(None, col, symptoms_byuser[i]).ratio()
            if(ratio!=0 and maxn<ratio):
                maxn=ratio
                symptom=col
        max_n.append(maxn*100)
        symptoms.append(symptom)
        st.write(symptom)
        st.write(maxn*100,"%")

    #st.write("લક્ષણો ડેટાસેટ સાથે મેળ ખાતા : ",symptoms,max_n)
    #st.write("લક્ષણો અને એની સંભાવનાઓ : ",symptoms,max_n)
      
        
    #મને પેટ નો દુખાવો થાય છે ,મને માથુ પણ દુખે છે 
    #મને ત્વચા પર ચકામા, થાક ,ખંજવાળ અને વધારે તાવ આવે છે
    #રોગ ડાયાબિટીસ અને ટાઇફોઇડ થયો છે
    #મને ગળામાં ખંજવાળ આઈ રહી છે અને ગળામાં પેચ જેવુ પણ લાગે છે


# In[101]:


if flag == 0:

    # making data frame
    data = pd.read_excel("Gujarati_Dimensionality_Reduction.xlsx")
    i=0
    a = []

    # iterating the columns (symptoms-133)
    for col in data.columns:
        a.append(col) 
        i=i+1

    a.pop() # Detele last column which is for diseases
    l=len(a)
    s = [0] * l # Generate empty array s of size 132

    #symptoms=input('લક્ષણો દાખલ કરો  : ')

    #st.write(type(s))
    import nltk
    symptoms_word=symptoms
    for i in symptoms_word:
        s[a.index(i)]=1

    s=np.array(s)
    s=s.reshape(-1,1)
    s.shape
    s=s.T

    alldiseases = []
    for i in symptoms_word: 
        for k in range (41):
            if data[i][k] == 1:    
                alldiseases.append(data['રોગ'][k])  # All the disease of entered input

    def countfreq(alldiseases):
        freq_diseases = dict()

        for elem in alldiseases:
            # If element exists in dict then increment its value 
            if elem in freq_diseases:
                freq_diseases[elem] += 1
            else:
                freq_diseases[elem] = 1    

        freq_diseases = { key:value for key, value in freq_diseases.items() if value >= len(symptoms_word)}
        # Returns a dict of duplicate elements and thier frequency count
        return freq_diseases

    freq_diseases = countfreq(alldiseases)   
    freq_diseases = list(freq_diseases.items())
    commondiseases = np.array(freq_diseases)

    data = pd.DataFrame(commondiseases)
    data=data[data.columns[:-1]]
    commondiseases=data.to_numpy()

    st.write()
    if len(commondiseases) != 0:
        st.write("તમને થઈ શકે તેવા રોગની સંભાવનાઓ : ")
        alldiseases=commondiseases
        st.write(alldiseases)   
    else:
        x = np.array(alldiseases) 
        alldiseases=np.unique(x)
        st.write("તમને થઈ શકે તેવા રોગની સંભાવનાઓ : ")
        st.write(alldiseases)

    #ખાંસી કફ થાક 
    #ત્વચા_પર_ચકામા થાક ખંજવાળ સુસ્તી વધારે_તાવ


# In[107]:


if flag == 0:
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    disease=data.iloc[:, -1]
    len_disease=len(disease)

    len_alldisease=len(alldiseases)
    Number=[];

    j=0
    for i in range(len_disease):
        if j != len_alldisease:
            if alldiseases[j] ==  disease[i]:
                Number.append(i)
                j=j+1
        i=i+1

    feature_row=data.loc[Number,:]
    feature_row=feature_row.to_numpy()

    features=[]
    for j in range(len_alldisease):
        if j !=len_alldisease:
            for i in range(128):
                if feature_row[j][i] == 1:
                    features.append(a[i])
                i=i+1

    for k in range(len(symptoms_word)):
        j=0
        for j in features:
            if symptoms_word[k] == j:
                features.remove(j);           
        k=k+1   

    features=np.unique(features)

    if features != []:
        st.write("આ અન્ય લક્ષણો છે : ")
        st.write(features)
        st.write()
        st.write("તમારાથી સંબંધિત અન્ય કોઈ લક્ષણો કૃપા કરીને દાખલ કરો હા-1 , નાં-0, અટકાવવા માટે-stop")
    st.write()

    #s=s.T
    #user_features=input('લક્ષણો દાખલ કરો  : ')
    #import nltk
    #usersymptoms=nltk.word_tokenize(user_features)

    #for i in usersymptoms:
    #    s[a.index(i)]=1

    #st.write(s,s.shape,type(s))

    lst = [] 
    n = len(features)
    s=s.T

    i = 0 
    for m in features:
        
        st.write(m)
        ele = st.text_input("Enter 1 or 0 or stop", key=str(i))
        i = i + 1
        if ele == "stop":
            break
        if ele != "0":
            s[a.index(m)]=1
        if ele!="1" and ele !="0":
            break


    # હા-0 , નાં-1  #મને શ્વાસ ચડી રહ્યો છે , મને સાઇનસ પર દબાણ થઈ રહ્યું છે
    # અસ્વસ્થતા કાટવાળું_ગળફામ ઠંડી વધારે_તાવ વાસ પરસેવો  છાતીનો_દુખાવો ઝડપી_ધબકારા ાં સતત_છીંક


# In[108]:
if flag == 0:
    s=s.T
    #st.write(s,s.shape,type(s))
    
    #st.write(feature_row,s)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    data_temp = pd.read_excel ("Gujarati_Training.xlsx")
    X = data_temp.iloc[:, :-2].values
    Y = data_temp['prognosis- પૂર્વસૂચન']
    
    y = labelencoder.fit_transform(Y)
    #st.write(y)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X, y)
    pr1= classifier.predict(s)
    R1=labelencoder.inverse_transform(pr1)
    
    data = pd.read_excel ("Gujarati_Dimensionality_Reduction.xlsx")
    disease_original=data.loc[data['રોગ'].isin(R1)]
    temp=disease_original.drop(columns=['રોગ'])
    disease_original = np.array(temp)
    s=np.array(s)
    
    count=0
    total=0

    for j in range(len(disease_original.T)):
        if (disease_original.T[j]==1):
            total=total+1

    #st.write(total)
    for i in range(len(disease_original)):
        if (s[i].any()==disease_original.T[i].any() and disease_original.T[i].any()==1):
            count=count+1
    prp=(count/total)*100;
    st.write("દાખલ કરેલા લક્ષણો પર થી ",R1," થવાની સંભાવના છે ")
    


# In[109]:


if flag == 0:
    import pandas as pd

    data2 = pd.read_excel("Gujarati_Dataset2.xlsx")

    Y = data2[data2.columns[0]].to_numpy()
    disease = Y.tolist()

    lendisease2=len(disease)

    columnnum=[]
    j=0
    for i in range(lendisease2):
        if R1 ==  disease[i]:
            columnnum.append(i)

    disease_row=data2.iloc[columnnum,:]
    disease_row=disease_row.to_numpy()
    st.write(disease_row.T[0][0])
    #st.write("\nશું તમે વર્ણન વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    #Description = st.text_input(" ")
    #if Description != '0':
    st.write(disease_row[0][1])

    #st.write("\nશું તમે લક્ષણો વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    #Description = st.text_input(" ", key="a")
    #if Description != '0':
    st.write(disease_row[0][2])

    #st.write("\nશું તમે ઘરેલું ઉપાય વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    #Description = st.text_input(" ", key="b")
    #if Description != '0':
    st.write(disease_row[0][3])

    #st.write("\nશું તમે સારવાર વિશે જાણવા માંગો છો ? (હા / નાં) : \n")
    #Description = st.text_input(" ", key="b")
    #if Description != '0':
    st.write(disease_row[0][4])

    # હા-1 , નાં-0

