import numpy as np
from sklearn.utils import shuffle

# function to split data indexes regarding ratio
def ratio_zero_one_index_func(one_labels_index, zero_labels_index, ratio_value):
    
    zero_labels_index_with_ratio = zero_labels_index[: (len(one_labels_index)* ratio_value)]

    print("len of ones: ", len(one_labels_index), 'len of zeros that we chose:', len(zero_labels_index_with_ratio))
    print('ratio= ', len(zero_labels_index_with_ratio)/len(one_labels_index) )
    return zero_labels_index_with_ratio

D = np.loadtxt('../CombTVir_Dataset/Similarity_Matrix_Drugs.txt')
T = np.loadtxt('../CombTVir_Dataset/Similarity_Matrix_Viruses.txt')
with open('../CombTVir_Dataset/Y.npy', 'rb') as f:
    Y = np.load(f)

with open('../CombTVir_Dataset/Xindex.npy', 'rb') as f:
    XIndex = np.load(f)

XIndex = np.array(XIndex)
# print('the lenth of train set before applying ratio: ', len(XIndex))
zero_labels_index = [XIndex[i] for i in range(len(Y)) if Y[i]==0 ]
one_labels_index = [XIndex[i] for i in range(len(Y)) if Y[i]==1]
#print(zero_labels_index)
#print(one_labels_index)


ratios = [3,5,10,100,500]
for ratio in ratios:
    # split the data regarding choosen value
    zero_labels_index_ratio = ratio_zero_one_index_func(one_labels_index,zero_labels_index, ratio )

    XIndex = one_labels_index + zero_labels_index_ratio
    Y =  np.array([1 for i in range(len(one_labels_index))] + [0 for i in range(len(zero_labels_index_ratio))])

    # print('the lenth of train indexes after applying ratio: ', len(XIndex))
    # print('the lenth of test indexes after applying ratio: ', len(Y))

    # shuffle before splitting the data
    XIndex, Y = shuffle(XIndex, Y)
    ratioed_X = []
    for tup in XIndex:
        #ratioed_X.append(np.concatenate((D[tup[0]], D[tup[1]], T[tup[2]])))
        ratioed_X.append(np.concatenate((D[tup[0]], D[tup[1]], T[tup[2]] )))

    # save splitted data
    np.savetxt('../CombTVir_Dataset/ML/XIndex_ratio'+str(ratio)+'.txt' , X = XIndex)
    np.savetxt('../CombTVir_Dataset/ML/Y_ratio'+str(ratio)+'.txt' , X = Y)
    np.savez_compressed('../CombTVir_Dataset/ML/' + str(ratio) + '-to-1 y.npz', np.array(Y))
    np.savez_compressed('../CombTVir_Dataset/ML/' + str(ratio) + '-to-1 X.npz', np.array(ratioed_X))
