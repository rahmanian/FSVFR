#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:07:51 2019

@author: rahmanian
"""
import numpy as np
import numpy.matlib as mb
import time
import pandas as pd
import os.path, sys
import csv
from scipy.io import arff, loadmat
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

def loadData(filename):
    print('Loading and preprocessing data...')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        Data = list(reader)
    Data = np.array(Data)
    Data = Data.transpose();
    X = Data[:,1:] # Only Features
    y = Data[:,0]  # Class labels
    X = X.astype(np.float)
    
    Means = (X.mean(axis=0)).transpose()
    M = mb.repmat(Means,np.size(X,0),1)
    
    Features = X - M
    Features[Features >= 0] = 1
    Features[Features < 0] = 0
    
    numFeatures = np.size(X,1)
    print('Data loaded.')
    return Features, numFeatures, y

# Partial Joint Entropy
def partial_joint_entropy(base, cluster_centers, end):
    FS = FeatureSelection()
    y = cluster_centers[:end+1,:]
    t = base[:end+1]
    X = np.column_stack((t, y))
    PJE = FS.joint_entropy_more(X)
    return PJE

   

# Average Mutual Information
def average_mutual_info(base, cluster_members, end, Features):
    FS = FeatureSelection()
    mi = 0.0
    for i in cluster_members:
        mi += FS.mutual_information(base[:end+1], Features[:end+1,i])
    return mi/len(cluster_members)

def partialMSU(base, cluster_members, end, Features):
    FS = FeatureSelection()
    y = Features[:end+1,:]
    t = base[:end+1]
    # print(t.shape, y.shape)
    XX = np.column_stack((t, y))
    # print(XX.shape)
    PMSU = FS.MSU(XX)
    return PMSU

# Average Redundancy function.
def average_redundancy(clusters, selected_size, SU_Mat):
    clusters_size = np.zeros(len(clusters), dtype=np.int16)
    selected_features = np.zeros(selected_size, dtype=np.int16) 
    for i, cl in enumerate(clusters):
        clusters_size[i] = len(cl)
    index = np.argsort(-clusters_size)
    for i in range(selected_features.shape[0]):
#        print(f'SS={selected_size}, i={i}, index[i]={index[i]}, len(cls)={len(clusters[24])}')
        AR = np.zeros(len(clusters[index[i]]))
        for j in range(len(clusters[index[i]])):
            others = list(range(0,j))+list(range(j+1,len(clusters[index[i]])))
            for k in others:
                AR[j] += SU_Mat[j,k]
            AR[j] = AR[j]/len(clusters[index[i]])
        select = np.argmax(AR)
        selected_features[i] = clusters[index[i]][select]
    return selected_features, index

def SU_dist(x, y):
    global SU
    if SU is None:
        FS = FeatureSelection()
        su = FS.symmetrical_uncertainty(x,y)
        #t = su if su > 0 else 10e-15
        return 1.0/(su+10e-15)
    else:
        if type(x) is np.ndarray:
            x = int(x[0])
            y = int(y[0])
        # print(f'x={x}, y={y}')
        su = SU[x,y]
        #t = su if su > 0 else 10e-15
        return 1.0/(su+10e-15)

def MI_dist(x, y):
    su = mutual_info_regression(x.reshape(-1,1), y)
    return 1.0/(su+10e-15)

SU = None
def kNN(X, f, _SU = None,k=3, disc = True):
    global SU
    SU = _SU
    if disc:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=SU_dist)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=MI_dist)
    # nbrs.fit(X.T)
    # distances, indices = nbrs.kneighbors((X[:,f].T))#.reshape(1,-1))
    # return indices
    
    n,m = X.shape
    # indx = np.arange(m,dtype=int)
    indx = np.array(f, dtype=int)
    nbrs.fit(indx.reshape(-1,1))
    distances, indices = nbrs.kneighbors(np.array(indx).reshape(-1,1))
    return indices

def check_same_values(X, thr=0.98):
    features = []
    m, n = X.shape
    for i in range(n):
        #print(len(f))
        v, c = np.unique(X[:,i], return_index=False, return_inverse=False, return_counts=True)
        r = c/np.sum(c)
        if np.sum(r>thr)>0:
            features.append(i)
    return np.array(features)

class FeatureSelection:
    def entropy(self,X):
        x_values, x_counts = np.unique(X, return_counts=True)
        N = X.shape[0] if X.size != 1 else 1
        p = np.array(x_counts)/float(N)
        log2_p = np.log2(p)
        hx = -np.sum(p*log2_p)
        return hx, p
    
    def conditional_entropy(self, X, y):
        hx, px = self.entropy(X)
        x_values, x_counts = np.unique(X, return_counts=True)
        hy = np.zeros(len(x_values),dtype=np.float)
        for i, value in enumerate(x_values):
            mask_v = X==value
            hy[i], _ = self.entropy(y[mask_v])
        return np.sum(px*hy), px, hy
    
    def joint_entropy_two(self, X, y):
        hx, px = self.entropy(X)
        hyx, _, _=self.conditional_entropy(X, y)
        return hx + hyx
    
    def joint_entropy_more(self, X):
        if X.shape[1] < 2:
            raise ValueError('Input must be have more than one feature.')
        values, counts = np.unique(X, return_index=False, return_inverse=False, return_counts=True, axis=0)
        prob = counts/X.shape[0]
        return -np.sum(prob*np.log2(prob))
    
    def mutual_information(self, X1, X2):
        hx2,_ = self.entropy(X2)
        hx2x1,_,_ = self.conditional_entropy(X1,X2)
        return hx2 - hx2x1#, hy, hyx
    
    def conditional_mutual_information(self, X, Y, Z):
        """
        compute:
            I(X;Y|Z)=H(X,Z)+H(Y,Z)-H(X,Y,Z)-H(Z)
        """
        Hz,_ = self.entropy(Z)
        Hxz = self.joint_entropy_two(X,Z)
        Hyz = self.joint_entropy_two(Y,Z)
        tmp = np.hstack((X.reshape(-1,1),Y.reshape(-1,1),
                         Z.reshape(-1,1)))
        Hxyz = self.joint_entropy_more(tmp)
        return Hxz + Hyz - Hxyz - Hz
    
    def MSU(self, X):
         X = X.T
         from pyitlib import discrete_random_variable as drv
         joint_H = drv.entropy_joint(X)
         H = [drv.entropy(X[i]) for i in range(X.shape[0])]
         n = len(H)
         f = n/(n-1)
         sum_H = np.sum(H)
         return f*(1-(joint_H/sum_H))

    def symmetrical_uncertainty(self, X1, X2):
        hx1,_ = self.entropy(X1)
        hx2,_ = self.entropy(X2)
        mi = self.mutual_information(X1, X2)
        return 2*mi/(hx1+hx2)

class featureClustering(FeatureSelection):
    def __init__(self, Debug=True):
        self.Debug = Debug
        self.name = ''
        self.numClass = 0
        self.data_set_names = {'ecoli':['data', 'ecoli.data'],
                               'thoracic':['arff', 'ThoraricSurgery.arff'],
                               'parkinsons':['data', 'parkinsons.data'],
                               'breast_cancer':['data', 'BreastCancer.data'],
                               'lung':['data', 'lung-cancer.data'],
                               'spambase':['data', 'spambase.data'],
                               'fertility':['data', 'fertility_Diagnosis.txt'],
                               'breast_tissue':['xls', 'BreastTissue.xls' ],
                               'image':['data', 'segmentation.data'],
                               'qsar':['data', 'QSAR.csv'],
                               'sonar':['data', 'sonar.all-data'],
                               'mfeat-fourier':['arff','image/dataset_14_mfeat-fourier.arff'],
                               'AR10P':['mat','image/warpAR10P.mat'],
                               'PIE10P':['mat','image/warpPIE10P.mat'],
                               'CLL_SUB_111':['mat','microarray/CLL_SUB_111.mat'],
                               'ALLAML':['mat','microarray/ALLAML.mat'],
                               'arrhythmia':['data','microarray/arrhythmia.data'],
                               'colon':['mat','microarray/colon.mat'],
                               'Embryonal_tumours':['arff','microarray/Embryonal_tumours.arff'],
                               'B-Cell1':['arff','microarray/B-Cell1.arff'],
                               'B-Cell2':['arff','microarray/B-Cell2.arff'],
                               'B-Cell3':['arff','microarray/B-Cell3.arff'],
                               'SMK_CAN_187':['mat','microarray/SMK_CAN_187.mat'],
                               'TOX_171':['mat','microarray/TOX_171.mat'],
                               'chess':['arff','text/chess.dat'],
                               'coil2000':['arff','text/coil2000.dat'],
                               'wine':['data', 'wine/wine2.data'],
                               'oh0.wc':['arff','text/oh0.wc.arff'],
                               'tr11.wc':['arff','text/tr11.wc.arff'],
                               'tr12.wc':['arff','text/tr12.wc.arff'],
                               'tr21.wc':['arff','text/tr21.wc.arff']}
    def load_data(self, data_set='ALLAML', base='ICCKE2020_dataset/'):
        try:
            [ds_type, ds_path] = self.data_set_names[data_set]   
            self.name = data_set
        except KeyError:
            print(f'Error: Dataset name <{data_set}> is not valid.')
            sys.exit(-1)
        else:
            if ds_type == 'arff':
                data = arff.loadarff(base+ds_path)
                df = pd.DataFrame(data[0])
                df = df.dropna(axis=1,how='all')
                df = df.fillna(df.mean())
                tmp =  (df.iloc[:,-1].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                self.X = df.iloc[:,:-1].to_numpy()
                if data_set == 'chess':
                    tmp = np.empty_like(self.X, dtype=np.int16)
                    tmp[self.X == b'f'] = 0
                    tmp[self.X == b'n'] = 0
                    tmp[self.X == b'l'] = 0
                    tmp[self.X == b't'] = 1
                    tmp[self.X == b'w'] = 1
                    tmp[self.X == b'g'] = 1
                    tmp[self.X == b'b'] = 2
                    self.X = np.copy(tmp)
            elif ds_type == 'mat':
                data = loadmat(base+ds_path)
                self.label = np.copy(data['Y'])
                self.X = np.copy(data['X'])
            elif ds_type == 'data':
                if data_set == 'parkinsons':
                    data = pd.read_csv(base+ds_path, header=0, index_col=0)
                elif data_set == 'breast_cancer':
                    data = pd.read_csv(base+ds_path, header=None, index_col=0)
                else:
                    data = pd.read_csv(base+ds_path, header=None)
                df = pd.DataFrame(data)
                df = df.dropna(axis=1,how='all')
                df = df.fillna(df.mean())
                if data_set == 'parkinsons':
                    tmp = (df['status'].to_numpy())
                elif data_set in ['breast_cancer','lung', 'image']:
                    tmp = (df.iloc[:,0].to_numpy())
                else:
                    tmp = (df.iloc[:,-1].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                if data_set == 'parkinsons':
                    self.X = df.drop('status', inplace=False, axis=1).to_numpy()
                elif data_set in ['lung','breast_cancer', 'image']:
                    self.X = df.iloc[:,1:].to_numpy()
                else:
                    self.X = df.iloc[:,:-1].to_numpy()
            elif ds_type == 'xls':
                data = pd.read_excel(base+ds_path, header=0, index_col=0, sheet_name='Data')
                df = pd.DataFrame(data)
                print(df.shape)
                tmp = (df.iloc[:,0].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                self.X = df.iloc[:,1:].to_numpy()
            else:
                print(f'Type of dataset <{ds_type}> is not valid.')
                return
            self.numFeatures = self.X.shape[1]
            if isinstance(self.label[0,0],bytes):
                try:
                    for i in range(self.label.shape[0]):
                        self.label[i,0] = int(self.label[i,0]) 
                except:
                    for i in range(self.label.shape[0]):
                        self.label[i,0] = self.label[i,0].decode('utf-8')
            self.name = data_set
            tmp = np.unique(self.label, return_index=False, return_inverse=False,
                          return_counts=False)
            self.numClass = len(tmp)
            self.uniqueLabel = tmp[:]
            
    def _binary(self):
        Means = self.X.mean(axis=0)
        M = mb.repmat(Means,np.size(self.X,0),1)
        
        self.BinX = self.X - M
        self.BinX[self.BinX >= 0] = 1
        self.BinX[self.BinX < 0] = 0
        
    def _discretization(self, k=2):
#        Means = self.X.mean(axis=0)
#        Stds = self.X.std(axis=0)
#        M = mb.repmat(Means,np.size(self.X,0),1)
#        
#        dMean = self.X - M 
#        dMean[dMean>Stds] = 1
#        dMean[dMean<-Stds] = -1
#        a = (dMean<Stds)
#        b = (dMean>-Stds)
#        dMean[a & b] = 0
#        self.TernX = np.copy(dMean)
        kDist = KBinsDiscretizer(k, encode='ordinal', strategy='uniform')
        self.DiscX = kDist.fit_transform(self.X).astype(np.int)
        
    def _remove_useless(self, thresh=0.98):
        useless_features = check_same_values(self.X, thr=thresh)
        if useless_features.size > 0:
            self.X = np.delete(self.X, useless_features, 1)

    def preprocess(self, disc=True, k=2, useless=True, thr=0.98):
        if useless:
            self._remove_useless(thresh=thr)
        if disc:
            self._discretization(k=k)
            # self.DiscX = self.DiscX - 1
        else:
            self.DiscX = np.copy(self.X)
        self.numFeatures = self.X.shape[1]
    
    def NMutInfo(self, path, recalculate=False):
        if self.Debug:
            print(f'Calculate Normalized Mutual Information for {self.name} dataset.')
        if  (not recalculate) and os.path.exists(path):
            self.NMI = np.load(path)
        else:
            import time
            start = time.time()
            self.NMI = np.zeros((self.numFeatures, self.numFeatures))
            for i in range(self.numFeatures):
                if self.Debug and i%300==0:
                    print(f'\t\tFeatures #{i} to {i+300} vs others.')
                for j in range(i+1, self.numFeatures):
                    self.NMI[i,j] = metrics.normalized_mutual_info_score(self.DiscX[:,i], self.DiscX[:,j])
                    self.NMI[j,i] = self.NMI[i,j]
            print(f"NMI time = {time.time()-start} seconds")
            np.save(path, self.NMI)
    
    def symmetric_uncertainty(self, path, recalculate=False, disc=True):
        FS = FeatureSelection()
        if self.Debug:
            print(f'Calculate Symmetric Uncertainty for {self.name} dataset.')
        if  (not recalculate) and os.path.exists(path):
            self.SU = np.load(path)
        else:
            import time
            start = time.time()
            self.SU = np.zeros((self.numFeatures, self.numFeatures))
            for i in range(self.numFeatures):
                if self.Debug and i%300==0:
                    print(f'\t\tFeatures #{i} to {i+300} vs others.')
                for j in range(i+1, self.numFeatures):
                    if disc:
                        self.SU[i,j] = FS.symmetrical_uncertainty(self.DiscX[:,i], self.DiscX[:,j])
                    else:
                        self.SU[i,j] = mutual_info_regression(self.DiscX[:,i].reshape(-1, 1), self.DiscX[:,j])
                    self.SU[j,i] = self.SU[i,j]
            print(f"SU time = {time.time()-start} seconds")
            np.save(path, self.SU)
            
    def KNN(self, path, ratio=1, recalculate=False, disc=True):
        if self.Debug:
            print(f'Calculate kNN for {self.name} dataset.')
            start = time.time()
        if  (not recalculate) and os.path.exists(path):
            self.kNN = np.load(path)
        else:
            self.k = int(np.sqrt(self.numFeatures)*ratio)
            self.kNN = kNN(self.DiscX, list(range(self.numFeatures)),self.SU, k=self.k+1, disc=disc)
            np.save(path, self.kNN)
        if self.Debug:
            end = time.time()
            print(f'\tTime for kNN on {self.name} is {end - start} seconds')
    
    def Density_kNN(self, D_kNN_filename=None, base_path='D_kNN'):
        self.k = int(np.sqrt(self.numFeatures))
        if D_kNN_filename and os.path.exists(D_kNN_filename):
            self.D_kNN = np.load(base_path+"/"+D_kNN_filename)
        else:
            self.D_kNN = np.zeros(self.numFeatures, dtype=np.float)
            for f in range(self.numFeatures):
                sum_of_SU = 0.0
                for neigbor in self.kNN[f,1:]:
                    sum_of_SU += self.SU[f,neigbor]
                self.D_kNN[f] = sum_of_SU/(self.k)
            name = "D_kNN_"+self.name+".npy" if D_kNN_filename == None else D_kNN_filename 
            np.save(f'{base_path}/{name}', self.D_kNN)
        
    def initial_centers(self):
        self.m = 0
        self.maxSU = -10e36
        sorted_D_kNN = np.argsort(-self.D_kNN)   
        
        Fs_ind = np.copy(sorted_D_kNN)
        Fs_ind = Fs_ind.astype(np.int16)
        self.Fc = np.copy(self.DiscX[:,Fs_ind[0]].reshape(self.DiscX.shape[0],1))
        self.Fc_ind = np.array([Fs_ind[0]], dtype=np.int16)
        Fs_ind = np.delete(Fs_ind, np.where(Fs_ind == self.Fc_ind))
        self.m += 1
        
        for fs in Fs_ind:
            tmpMax= []
            for fc in self.Fc_ind:
                tmpMax += [max(self.maxSU, self.SU[fs,fc])]
                if (fc in self.kNN[fs]) or (fs in self.kNN[fc]):
                    break
            else:
                    self.Fc_ind = np.append(self.Fc_ind, [fs])
                    self.Fc = np.hstack([self.Fc, self.DiscX[:,fs].reshape(self.DiscX.shape[0],1)])
                    self.m += 1
                    self.maxSU = np.max(tmpMax)
        print(f'init: m={self.m}')
        
    def main_loop(self, path, max_feature=100, MAX_ITER=20, recalculate=False):
        self.Fc_ind = self.Fc_ind[:max_feature]
        self.Fc = self.Fc[:,:max_feature]
        if max_feature>self.Fc.shape[1]:
            max_feature = self.Fc.shape[1]
        
        print('main loop of my algorithm...')
        if (not recalculate) and os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            clusters = np.copy(data['clusters'])
            self.Fc = np.copy(data['fc'])
        else:
            FS = FeatureSelection()
            changed = True
            
            itr = 1
            while changed and itr < MAX_ITER:
                print(max_feature, self.Fc.shape[1])
                assert(max_feature == self.Fc.shape[1])
                print(f'Assign samples ...#{itr}, cent={len(self.Fc_ind)}')
                Fs = np.copy(self.DiscX)
                finished = False
                while not finished:
            #        if m > numFeatures:
            #            break
                    clusters = [[None] for i in range(self.Fc.shape[1])]
                    for k, fs in enumerate(Fs.T):
                        if k%100==0:
                            pass#print(f'Iter#{itr}\t Feature#{k}')
                        SU_centers = np.zeros(self.Fc.shape[1], dtype=np.float)
                        for i, fc in enumerate(self.Fc.T):
                            tmp = FS.symmetrical_uncertainty(fs, fc)
                            # tmp = FS.mutual_information(fs, fc)
                            SU_centers[i] = tmp
                        j = np.argmax(SU_centers)
                        clusters[j].append(k)
                    else:
                        finished = True
                print('Samples Assigned.')
                for i in range(self.Fc.shape[1]):
                    clusters[i] = clusters[i][1:]
                print(f'Select center of clusters...#{itr}')
                new_Fc = np.copy(self.Fc)
                change_Fc = False
                for i in range(self.Fc.shape[1]):
                    fc = np.copy(self.Fc[:,i]) 
                    tmp_fc = np.copy(fc)
                    for val in range(len(fc)):
                        mi_1 = average_mutual_info(fc, clusters[i], val, self.DiscX) 
                        #mi_1 = partialMSU(fc, clusters[i], val, self.DiscX) 
                        #pje_1 = partial_joint_entropy(fc, Fc, val)
                        fc[val] = 1 - fc[val]
                        mi_2 = average_mutual_info(fc, clusters[i], val, self.DiscX) 
                        #mi_2 = partialMSU(fc, clusters[i], val, self.DiscX) 
                        #pje_2 = partial_joint_entropy(fc, Fc, val)
                        #if (mi_2 < mi_1 and pje_2 < pje_1) or (mi_2<mi_1 and pje_2>pje_1) or (mi_2>mi_1 and pje_2<pje_1):
                        #    fc[val] = 1-fc[val]
                        fc[val] = fc[val] if mi_2>mi_1 else 1-fc[val] 
                    tmp = np.array(fc).reshape(len(fc),1)
                    new_Fc[:,i] = tmp[:,0]
                    if np.any(fc != tmp_fc):
                        change_Fc = True
                if not change_Fc:
                    changed = False
                self.Fc = new_Fc.copy()
                print('Center of clusters selected.')
                itr += 1
            np.savez(path, clusters=clusters, fc=self.Fc)
        print('main loop of my algorithm finished.')
        
        self.selected_ = []
        Fs = np.copy(self.DiscX)
        for j, fc in enumerate(self.Fc.T):
            print(f'cluster #{j} ==> {len(clusters[j])}')
            mi = np.zeros(len(clusters[j]))
            for k,i in enumerate(clusters[j]):
                mi[k] = self.mutual_information(fc, Fs[:,i])
            self.selected_.append(clusters[j][np.argmax(mi)])
    
    def _learner(self, full=False):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.numClass, init='k-means++', random_state=0)
        if not full:
            X = self.DiscX[:,self.selected_]
        else:
            X = self.DiscX[:,:]
        kmeans.fit(X)
        y_hat = np.copy(kmeans.labels_)
        y_hat = y_hat.reshape(-1,1)
        return y_hat   
    
    def compute_scores(self):
        from sklearn.metrics import f1_score, adjusted_mutual_info_score
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        y_hat_full = self._learner(full = True)
        y_hat_partial = self._learner(full = False)
        y_hat_full = y_hat_full.reshape(y_hat_full.shape[0],)
        y_hat_partial = y_hat_partial.reshape(y_hat_full.shape[0],)
        
        y_uniq = np.unique(self.label)
        y = np.zeros(self.label.shape[0])
        for i, _y in enumerate(y_uniq):
            tmp = self.label.T == _y
            y[tmp[0]] = i
                
        f_score_full = f1_score(y, y_hat_full, average = 'micro')
        f_score_part = f1_score(y, y_hat_partial,average = 'micro')
        
        ami_full = adjusted_mutual_info_score(y, y_hat_full)
        ami_part = adjusted_mutual_info_score(y, y_hat_partial)
        
        ars_full = adjusted_rand_score(y, y_hat_full)
        ars_part = adjusted_rand_score(y, y_hat_partial)
        
        silh_full = silhouette_score(self.DiscX, y_hat_full)
        silh_part = silhouette_score(self.DiscX[:,self.selected_], y_hat_partial)
                
        
        return f_score_full, f_score_part, ami_full, ami_part, ars_full, ars_part, silh_full, silh_part
    
    def computing_accuracy(self,path, recalculate=False):
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        import matplotlib.pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
        y = self.label.reshape(1,-1)[0]
        if isinstance(y[0], int):
            y = y.astype(np.int)            
        knn = KNeighborsClassifier(n_neighbors=3)
        cross = cross_val_score(knn, self.X, y, cv=5, scoring='accuracy')
        knn_all = cross.mean()
        knn_all_std = cross.std()
        print(f'Accuracy(knn) with all features = {knn_all:0.4f}.')
        
        svm_clf = svm.SVC()
        cross = cross_val_score(svm_clf, self.X, y, cv=5, scoring='accuracy')
        svm_all = cross.mean()
        svm_all_std = cross.std()
        print(f'Accuracy(svm) with all features = {svm_all:0.4f}.')
        knn_selected = 0.0
        svm_selected = 0.0
        if  (not recalculate) and os.path.exists(path):
            acc = np.load(path)
            knn_all = np.copy(acc['knn_all'])
            knn_selected = np.copy(acc['knn'])
            knn_selected_std = np.copy(acc['knn_selected_std'])
            svm_all = np.copy(acc['svm_all'])
            svm_all_std = np.copy(acc['smv_all_std'])
            svm_selected = np.copy(acc['svm'])
            svm_selected_std = np.copy(acc['svm_selected_std'])
        else:
                       
            num_test = 100 if self.numFeatures>100 else self.numFeatures//2
            
            for selected_size in range(len(self.selected_),len(self.selected_)+1):
                #selected_features, index = average_redundancy(clusters, selected_size, SU_Mat)
                ss = self.X[:,self.selected_]
                tmp = cross_val_score(knn, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                knn_selected = tmp.mean()
                knn_selected_std = tmp.std()
                print(f'Accuracy(knn) with {selected_size} select features = {knn_selected:0.4f}.')
                
                tmp = cross_val_score(svm_clf, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                svm_selected = tmp.mean()
                svm_selected_std = tmp.std()
                print(f'Accuracy(svm) with {selected_size} select features = {svm_selected:0.4f}.')

            np.savez(path, knn_all=knn_all, knn=knn_selected, svm_all=svm_all, svm=svm_selected)
        
        res = (knn_all, knn_all_std, knn_selected, knn_selected_std)
        res += (svm_all, svm_all_std, svm_selected, svm_selected_std)
        return res
