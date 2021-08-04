#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:22:50 2019

@author: rahmanian
"""


from FeatureSelectionModule import featureClustering


fc = featureClustering(Debug=True)
base = 'dataset/'
names = ['AR10P','arrhythmia','chess','coil2000','colon','mfeat-fourier','PIE10P','TOX_171', 'wine']
r = 1
if __name__ == '__main__':        
    
    for name in names:
        if name == 'wine': # running algorithm only for wine dataset
            print(f'loading and preprocessing for {name} dataset...')
            fc.load_data(name, base=base)
            fc.preprocess(disc=True,k=2)
            print(f'{name} dataset with shape={fc.DiscX.shape} loaded.')

            print(f'Computting SU for {name} dataset...')
            fc.symmetric_uncertainty(path=f'SU_Mat/{name}_SU.npy', recalculate=True, disc=True)
            print(f'SU for {name} dataset computed.')

            print(f'Computting kNN for {name} dataset...')
            fc.KNN(path=f'KNN/KNN_{name}_{r}.npy', ratio=1, recalculate=True, disc=True)
            print(f'kNN for {name} dataset computed.')

            print(f'Computting D_kNN for {name} dataset...')
            fc.Density_kNN(f'D_kNN_{name}_std_{r}.npy',base_path="D_kNN")
            print(f'D_kNN for {name} dataset computed.')

            print(f'Select initial centers for {name} dataset...')
            fc.initial_centers()
            print('Initial centers selected.')

            # alg_type = 'FSFC',... 
            alg_type = 'MSU' # our method
            print(f'Main Loop of {alg_type} algorithm...')
            max_feat = 100 if fc.numFeatures>100 else fc.numFeatures//2
                 
            if alg_type == 'MSU':
                    fc.main_loop(path=f'clusters/{alg_type}_{name}_{r}.npz',max_feature=fc.numFeatures, 
                              recalculate=True, MAX_ITER=20)
            else:
                    pass# implement other methods
            fs_f, fs_p, ami_f, ami_p, ars_f, ars_p, sil_f, sil_p = fc.compute_scores()
            print(f'f_score_full={fs_f:0.4f}, '
                  f'f_score_part={fs_p:0.4f}\n'
                  f'NMI_full    ={ami_f:0.4f}, '
                  f'NMI_part    ={ami_p:0.4f}\n'
                  f'ARI_full    ={ars_f:0.4f}, '
                  f'ARI_part    ={ars_p:0.4f}\n'
                  f'SILH_full   ={sil_f:0.4f}, '
                  f'SILH_part   ={sil_p:0.4f}\n')
            print(fc.selected_)
            with open(f'Accuracy/{alg_type}_accuracy_{name}_{r}.txt','w') as f:
                    f.write(f'f_score_full={fs_f:0.4f},'
                            f'f_score_part={fs_p:0.4f}\n'
                            f'NMI_full    ={ami_f:0.4f},'
                            f'NMI_part    ={ami_p:0.4f}\n'
                            f'ARI_full    ={ars_f:0.4f},'
                            f'ARI_part    ={ars_p:0.4f}\n'
                            f'SILH_full   ={sil_f:0.4f},'
                            f'SILH_part   ={sil_p:0.4f}\n'
                            f'features: {fc.selected_}\n'
                            f'n#feature   ={len(fc.selected_)}')
            knn_full, knn_f_std, knn, knn_s_std, svm_full, svm_f_std, svm,svm_s_std = fc.computing_accuracy(f"Accuracy/{alg_type}_accuracy_{name}.npz", recalculate=True)
            with open(f'Accuracy/{alg_type}_accuracy_{name}_{r}.txt','a') as file:
                    file.write('\n')
                    file.write(f'knn_full \t= {knn_full}({knn_f_std})\n'
                               f'knn_selected = {knn}({knn_s_std})\n'
                               f'svm_full \t= {svm_full}({svm_f_std})\n'
                               f'svm_selected = {svm}({svm_s_std})\n')       
