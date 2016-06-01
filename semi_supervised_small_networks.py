
# coding: utf-8

# In[1]:

import networkx as nx
import numpy as np
import os
import codecs
import json
from collections import defaultdict as dd


# In[2]:

def get_user_network(u_id):
    p = r'D:\University\Mishenin\ids\\' + str(u_id) + '\user_network.txt'
    with codecs.open(p, 'r','utf-8') as inf:
        netw = json.load(inf)
    inf.close()

    ids = netw.keys()

    for k, wrong_id in netw.items():
        wr_id = []
        for f in wrong_id:
            if str(f) not in ids:
                wr_id.append(f)
        netw[k] = filter(lambda x: x not in wr_id,netw[k])  
              
    return netw


# In[ ]:

#user_id = 1019809
user_network = get_user_network(user_id)
friends = len(user_network.keys())


# In[28]:

from sklearn.externals import joblib
log_regression = joblib.load('D:/test/models/log_regression/log_regression.pkl')
KNN = joblib.load('D:/test/models/KNN/knn.pkl')
RFC = joblib.load('D:/test/models/RFC/rfc.pkl')


# In[4]:

import datetime as dt
year = dt.datetime.now().year

def map_age(x):
    if x is not None:
        splited = x.split('.')
        if len(splited) == 3:
            return year - int(splited[2])
    return None


# In[5]:

class my_map(object):

    def __init__(self, network):
        self.num_of_id = dict(enumerate(network.adj.keys()))
        self.id_of_num = dict(zip( network.adj.keys(), range(len(network.adj.keys()))))

    def get_id(self, num):
        return self.num_of_id[num]
    def get_num(self, id):
        return self.id_of_num[id]
    def ids(self):
        return self.id_of_num.keys()
    def nums(self):
        return self.num_of_id.keys()

    def __repr__(self):
        return self.num_of_id.__repr__() + "\n\n" + self.id_of_num.__repr__()


# In[6]:

list_of_attr = ['sex', 'city', 'country', 'graduation', 'university', 'school']


# In[7]:

def get_age_from_file(id_us):
    #тут считывать значение возраста из файла в директории r'F:\test\id' + numb_folder + '\\' + str(id_us) + '.txt'
    p = r'D:\University\Mishenin\ids\\' + str(user_id) + '\\' + str(id_us) + '.txt'
    with codecs.open(p, 'r','utf-8') as inf:
        inf_about_user = json.load(inf)
    inf.close()
    
    if 'bdate' not in inf_about_user:
        return None
    else:
        return map_age(inf_about_user['bdate'])


# In[8]:

def find_w(id_1, id_2, m_fr, n_method):
    features = []
    features.append(get_feature(id_1, id_2, m_fr))
    if n_method == 1:
        w = log_regression.predict_proba(features)[0][1]
    if n_method == 2:
        w =  KNN.predict_proba(features)[0][1]
    if n_method == 3:
        w = RFC.predict_proba(features)[0][1]
    
    return w


# In[9]:

def get_feature(id_user1, id_user2, m_fr):
    res_feat = []
    res_feat = comparing(get_inf_from_file(id_user1), get_inf_from_file(id_user2), m_fr)
    return res_feat


# In[10]:

def mutual_friends(m_fr):
    if friends <= 0:
        return 0
    else:
        return m_fr / float(friends)


# In[11]:

def comparing(inf1, inf2, m_fr):
    res = []
    res.append(mutual_friends(m_fr))
    for i in range(1, len(list_of_attr) + 1):
        res.append(comparing_components(inf1[i], inf2[i]))
    res.append(res[5] and res[6])
    return res


# In[12]:

def comparing_components(comp_1, comp_2):
    if comp_1 == None or comp_2 == None:
        return 0
    else:
        return int(comp_1 == comp_2)


# In[13]:

def get_inf_from_file(id_us):
    p = r'D:\University\Mishenin\ids\\' + str(user_id) + '\\' + str(id_us) + '.txt'
    with codecs.open(p, 'r','utf-8') as inf:
        inf_about_user = json.load(inf)
    inf.close()
    inf_about_user = dd(lambda: None, inf_about_user)
    res_inf = []
    res_inf.append(inf_about_user['id'])
    res_inf.append(inf_about_user['sex'])
    try:
        res_inf.append(inf_about_user['city']['id'])
    except:
        res_inf.append(None)
    try:
        res_inf.append(inf_about_user['country']['id'])
    except:
        res_inf.append(None)
    res_inf.append(inf_about_user['graduation'])
    res_inf.append(inf_about_user['university'])
    try:
        res_inf.append(int(inf_about_user['schools'][0]['id']))
    except:
        res_inf.append(None)
    #print res_inf
    return res_inf


# In[45]:

def init_graph(u_network):
    graph = nx.Graph()

    all_friend = u_network.keys()
    
    for k, v in u_network.items():
        for t in v:
            e = 0
            for el in u_network[str(t)]:
                if el in v:
                    e += 1
            w = find_w(k, t, e, 1)
            graph.add_edge(int(k),int(t), weight = w)
    
    #for u in all_friend: 
    #    if int(u) == user_id:
    #        continue
    #    graph.add_edge(int(u), user_id, weight = 1)
    #    for v in all_friend:
    #        if int(v) == user_id:
    #            continue
    #        if v != u:
    #            #матрица смежности
    #            if int(v) in u_network[u]:
    #                w = 1
    #            else:
    #                w = 0
    #            graph.add_edge(int(u),int(v), weight = w)
                
                
    #            #машинное обучение
    #            if v in u_network[u]:
    #                print mi
    #                e = 0
    #                for el in u_network[u]:
    #                    if el in u_network[v]:
    #                        e += 1
    #                w = find_w(u, v, e, 1)
    #                graph.add_edge(int(u),int(v), weight = w)
                
    return graph


# In[142]:

d = nx.Graph()
d.add_edge(1, 2, weight = 1)
d.add_edge(1, 3, weight = 1)
d.add_edge(1, 4, weight = 1)
d.add_edge(2, 4, weight = 1)
dd = my_map(d)
d.adj


# In[15]:

def get_class(age, k):
    if k == 20:
        return age / 5
    if k == 100:
        return age
    print 'Error_in_Get_class'


# In[16]:

def power(a, p):
    if p < 0:
        return np.linalg.inv(a) ** abs(p)
    elif p == 0:
        return np.eye(a.shape[0])
    else:
        return a ** p


# In[17]:

def find_A(id_num, us_network): #матрица W
    n = len(id_num.nums())
    A = np.zeros([n, n])
    
    for k, v in us_network.adj.items():
        i = id_num.get_num(k)
        for h, w in v.items():
            j = id_num.get_num(h)
            A[i, j] = A[j, i] = w['weight']
    
    return A


# In[18]:

def find_classes(id_num, k, m):
    classes = {}
    i = 0
    check = []
    for _id in id_num.ids():
        # получение возраста
        age = get_age_from_file(_id) #функция в Create sample for machine learning
        if age:
            i += 1
            
            if i % 10 == 0:
                check.append(_id)
                classes[id_num.get_num(_id)] = -1
                continue
                
            if i % m == 0:
                classes[id_num.get_num(_id)] = -1
                continue
                
            _class = get_class(age, k)
            if _class > k - 1:
                _class = -1
        else:
            _class = -1

        classes[id_num.get_num(_id)] = _class
    return (classes, check)

def get_Y(C, k):
    n = len(C)
    Y = np.zeros((n,k))

    for key, value in C.items():
        if value > -1:
            Y[key][value] = 1

    return Y


def classification(graph, sigma = 1, alpha = 0.75, k = 100):
    res = []
    mu =  2 / alpha - 2
    id_num = my_map(graph)
    A = find_A(id_num, graph)

    C, check = find_classes(id_num, k, np.random.randint(5, 15))
    for j in range(0,5):        
        C, check = find_classes(id_num, k, np.random.randint(5, 15))
        Y = get_Y(C, k)
        D = np.diag(np.sum(A, axis = 1))
    
        I = np.eye(len(A))
    
        F = np.zeros(Y.shape)
    
    #TMP_D = power(D, (-1) * sigma)
    #TMP_1 = alpha * np.dot(A, TMP_D)
    #TMP_2 = (1 - alpha) * np.linalg.inv(I - TMP_1)
    #
    #for x in range(k):
    #    F[:, x] = np.dot(TMP_2, Y[:, x])
    
    #общая формула
        TMP_D_1 = power(D, (-1) * sigma)
        TMP_D_2 = power(D, sigma - 1)
        TMP_1 = np.dot(TMP_D_1, A)
        TMP_2 = np.dot(TMP_1, TMP_D_2)
        TMP_3 = np.linalg.inv(I - alpha * TMP_2)
        for x in range(k):
            F[:, x] = (1 - alpha) * np.dot(TMP_3, Y[:, x])
    
        ANSWER = np.argmax(F, axis = 1)
    
        L = id_num.id_of_num.copy()
        for key,value in L.items():
            L[key] = str(ANSWER[value])

        res.append((L, check))
    return res


# In[46]:

id_users = os.listdir(r'D:\University\Mishenin\ids\\')
id_users[526]


# In[32]:

def ch(user_id):
    global friends
    #for x in id_users:
    #user_id = int(x)
    user_network = get_user_network(user_id)
    friends = len(user_network.keys())
    #if friends > 200:
    #    continue
    s_graph = nx.Graph()
    s_graph = init_graph(user_network)
    ans = classification(s_graph)
    ans = list(ans)
    check = ans[0][1]
    answ = []
    for i in ans:
        answ.append(i[0])
    
    return answ


# In[47]:

user_id = 1007070

get_ipython().magic(u'time ch(user_id)')


# In[19]:

#id_users = os.listdir(r'D:\University\Mishenin\ids\\')
id_users = [1007070]

for x in id_users:
    user_id = int(x)
    user_network = get_user_network(user_id)
    friends = len(user_network.keys())
    #if friends > 200:
    #    continue
    s_graph = nx.Graph()
    s_graph = init_graph(user_network)
    ans = classification(s_graph)
    ans = list(ans)
    check = ans[0][1]
    answ = []
    for i in ans:
        answ.append(i[0])
    
    #err = statict(answ, check)
    #save_2(err)
    #save_user(answ)


# In[43]:

def mi():
    for i in range(5):
        yield (i, 2)

t = mi()
t = list(t)
a = []
for i in t:
    a.append(i[0])
    
a


# In[20]:

s_graph = nx.Graph()
s_graph = init_graph(user_network)


# In[21]:

ans, check, W, id_num = classification(s_graph)


# In[20]:

def statict(answer, ch):
    stat_of_age = []
    for i in ch:
        s = 0
        for j in range(len(answer)):
            s += int(answer[j][i])
        s = s/float(len(answer))
        stat_of_age.append(abs(int(s) - int(get_class(get_age_from_file(i), 100))))
    summ = 0
    for i in range(len(stat_of_age)):
        summ = stat_of_age[i]**2 + summ
        
    save(stat_of_age)

    return (summ/float(len(stat_of_age)))**0.5


# In[21]:

def save(stat_of_age):
    pat = r'D:\University\Mishenin\err_adj\\' + str(user_id) + '_friends.txt'
    with open(pat, 'w') as ouf:
        for i in range(len(stat_of_age)):
            ouf.write("%f\n"%(stat_of_age[i]))


# In[22]:

def save_2(error):
    pat = r'D:\University\Mishenin\err_2_adj.txt'
    with open(pat, 'a') as ouf:
        ouf.write("%f\n"%(error))


# In[23]:

def save_user(answer):
    pat = r'D:\University\Mishenin\err_user_adj.csv'
    age = 0
    s = 0
    for j in range(len(answer)):
        s += int(answer[j][user_id])
    s = s/float(len(answer))
    age = (abs(int(s) - int(get_class(get_age_from_file(user_id), 100))))
    with open(pat, 'a') as ouf:
        ouf.write("%d\t%f\n"%(user_id, age))


# In[ ]:

tr = 0
fl = 0
for i in check:
    if str(ans[i]) == str(get_class(get_age_from_file(i), 100)):
        tr += 1
        print 'tr'
        print ans[i]
        print get_age_from_file(i)
        print '_________'
    else:
        fl += 1
        print 'fl'
        print ans[i]
        print get_age_from_file(i)
        print '_________'
print tr
print fl
print len(check)


# In[23]:

check


# In[ ]:



