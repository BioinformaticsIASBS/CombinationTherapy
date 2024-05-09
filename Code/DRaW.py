# requirements
import os
import argparse
import sys
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
from scipy import spatial
import tensorflow as tf
from tensorflow.keras import layers, Model
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, RocCurveDisplay
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, recall_score, average_precision_score, \
    confusion_matrix
from sklearn.utils import shuffle
from classifier_evaluation_DRaW import evaluate, ROC_curve, PR_curve, threshold_tuning


def train_DRaW_model(path, ratio, result_path):

  # drug
  mixed = np.loadtxt('../Data/Similarity_Matrix_Drugs.txt')

  # virus
  mixed_2 = np.loadtxt('../Data/Similarity_Matrix_Viruses.txt')


  with open('../Data/Y.npy', 'rb') as f:
      Y = np.load(f)

  with open('../Data/Xindex.npy', 'rb') as f:
      XIndex = np.load(f)

  XIndex = np.array(XIndex)
  # print('the lenth of train set before applying ratio: ', len(XIndex))

  # function to split data indexes regarding ratio
  def ratio_zero_one_index_func(one_labels_index, zero_labels_index, ratio_value):
      zero_labels_index_with_ratio = zero_labels_index[: (len(one_labels_index)* ratio_value)]

      print("len of ones: ", len(one_labels_index), 'len of zeros that we chose:', len(zero_labels_index_with_ratio))
      print('ratio= ', len(zero_labels_index_with_ratio)/len(one_labels_index) )
      return zero_labels_index_with_ratio

  zero_labels_index = [XIndex[i] for i in range(len(Y)) if Y[i]==0 ]
  one_labels_index = [XIndex[i] for i in range(len(Y)) if Y[i]==1]

  # split the data regarding choosen value
  zero_labels_index_ratio = ratio_zero_one_index_func(one_labels_index,zero_labels_index, ratio )

  XIndex = one_labels_index + zero_labels_index_ratio
  Y =  np.array([1 for i in range(len(one_labels_index))] + [0 for i in range(len(zero_labels_index_ratio))])

  # print('the lenth of train indexes after applying ratio: ', len(XIndex))
  # print('the lenth of test indexes after applying ratio: ', len(Y))

  # shuffle before splitting the data
  XIndex, Y = shuffle(XIndex, Y)

  # save splitted data
  np.savetxt(fname='../Data/'+path+'/XIndex_ratio'+str(ratio)+'.txt' , X = XIndex)
  np.savetxt(fname='../Data/'+path+'/Y_ratio'+str(ratio)+'.txt' , X = Y)

  def buildModule(inputShape,  output_bias=None):
      drop_out_rate = 0.5
      inputLayer = layers.Input(shape = inputShape)
      
      x = layers.Conv1D(128,3, activation = 'relu')(inputLayer)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Conv1D(64,3, activation = 'relu')(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Conv1D(32,3, activation = 'relu')(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Flatten()(x)
      
      x = layers.Dense(128, activation = 'relu')(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Dense(1, activation = 'sigmoid', bias_initializer=tf.keras.initializers.Constant(output_bias))(x)
      

      model = Model(inputLayer, x)
      # optimizer = tf.keras.optimizers.Adam()
      METRICS = [
        
  #      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  #      tf.keras.metrics.Precision(name='precision'),
  #      tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      ]
    # optimizer = get_optimizer()
      model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

      return model

  # K-fold Cross Validator 
  kfold = StratifiedKFold(n_splits = 10, shuffle=True, )
  outer_fold_no = 1

  auc_roc_test_list = []
  directory = '../'+'DRaW_results/'+'fold_results/' + str(ratio)
  os.mkdir(directory)

  outer_fold_metrics = [[],[],[],[],[],[],[],[]]
  test_metrics = [[],[],[],[],[],[],[],[]]

  for train_index, test_index in kfold.split(XIndex, Y):
    directory2 = directory + '/outer fold' + str(outer_fold_no)
    os.mkdir(directory2)


    
    X_testIndex= np.array( [XIndex[i] for i in test_index] )
    Y_test =  np.array([Y[i] for i in test_index] )
    np.savetxt(directory2 + '/test_indices.txt', X_testIndex)
    np.savetxt(directory2 + '/y_test.txt', Y_test)



    X_trainIndex= np.array([XIndex[i] for i in train_index])
    Y_train = np.array([Y[i] for i in train_index])
    
    np.savetxt(directory2 + '/train_indices.txt', X_trainIndex)
    np.savetxt(directory2 + '/y_train.txt', Y_train)


    evaluation_kfold = StratifiedKFold(n_splits = 5, shuffle=True, )

    inner_fold_no = 1

    
    
    inner_fold_metrcis = [[],[],[],[],[],[],[],[]]

    for train_prime_index, evaluation_index in evaluation_kfold.split(X_trainIndex, Y_train):

        directory3 = directory2 + '/ inner fold' + str(inner_fold_no)
        os.mkdir(directory3)

        X_prim_trainIndex= np.array([X_trainIndex[i] for i in train_prime_index])
        Y_prim_train = np.array([Y_train[i] for i in train_prime_index])

        X_evalIndex= np.array( [X_trainIndex[i] for i in evaluation_index] )
        Y_eval =  np.array([Y_train[i] for i in evaluation_index] )

        np.savetxt(directory3 + '/train_indices.txt', X_prim_trainIndex)
        np.savetxt(directory3 + '/y_train.txt', Y_prim_train)
        np.savetxt(directory3 + '/eval_indices.txt', X_evalIndex)
        np.savetxt(directory3 + '/y_eval.txt', Y_eval)
    
    # X_testIndex  = XIndex[test_index]
    # Y_test = Y[test_index]
    # X_trainIndex= XIndex[train_index]
    # Y_train = Y[train_index]


        # count number of negative samples
        neg_counter = 0
        for i in range(len(Y_prim_train)):
            if Y_prim_train[i] == 0:
                neg_counter +=1

        #number of negative and positive samples
        neg = neg_counter
        pos = len(Y_prim_train) - neg
        initial_bias = np.log([pos/ neg])

        
        model = buildModule((466,1), initial_bias)


        batch_size = 64
        epochs = 10

        # fold number 
        print('------------------------------------------------------------------------')
        print('outer fold ' + str(outer_fold_no) + ' inner fold ' + str(inner_fold_no) + ' ... ')

        # initial value for maximom aupr
        max_aupr = 0
        best_model = model
        
        for epoch in range(epochs):
            
            
            lossList = []
            iterates = len(Y_prim_train) // batch_size
            
            

            for it in range(iterates + 1):

                if it < iterates:
                    indexRange = [it * batch_size, (it + 1) * batch_size]
                else:
                    indexRange = [it * batch_size, len(Y_prim_train)]
                tempX = []
                for cIT in range(indexRange[0], indexRange[1]):

                    index = X_prim_trainIndex[cIT]
                    
                    #Concat DrugSim(index[0]) and ProteinSim(index[1]) => tempX
            
                    i_th_drug_index = index[0]
                    k_th_drug_index = index[1]
                    j_th_virus_index = index[2]

                    #i_th_drug= similarity_matrix_drug_disease[i_th_drug_index]
                    i_th_drug= mixed[i_th_drug_index]
                    k_th_drug = mixed[k_th_drug_index]
                    j_th_virus = mixed_2[j_th_virus_index]
                    
                    tempX.append(np.concatenate((i_th_drug ,k_th_drug, j_th_virus )))

                
                #print('shape of tempx:' , np.shape(tempX))

                tempX = tf.expand_dims(tempX, -1)
                    
                loss = model.train_on_batch(
                    tempX,
                    Y_prim_train[it * batch_size:(it + 1) * batch_size]
                )

                lossList.append(
                    loss
                )
                
            print('Train:')
            print('epoch ', epoch)
            lossList_0 = [lossList[i][0] for i in range(len(lossList))]
            lossList_1 = [lossList[i][1] for i in range(len(lossList))]
            lossList_5 = [lossList[i][2] for i in range(len(lossList))]
            
            # as I made the metrics less
            print('loss:', np.mean(lossList_0), 'auc:',np.mean(lossList_1), 'aupr:',np.mean(lossList_5))
            print('evaluation:')

            testIT = len(X_evalIndex) // batch_size
            #print(len(X_testIndex))
            testRes = []
            testY = []
            for it in range(testIT + 1):
                if it < testIT:
                    indexRange = [it * batch_size, (it + 1) * batch_size]
                else:
                    indexRange = [it * batch_size, len(X_evalIndex)]
                #print(indexRange)
                tempX = []
                for cIT in range(indexRange[0], indexRange[1]):
                    index = X_evalIndex[cIT]
                    testY.append(
                        Y_eval[cIT]
                    )
                    # Concat DrugSim(index[0]) and ProteinSim(index[1]) => tempX
            
                    i_th_drug_index = index[0]
                    k_th_drug_index = index[1]
                    j_th_virus_index = index[2]
                    
                

                    #i_th_drug= similarity_matrix_drug_disease[i_th_drug_index]
                    i_th_drug= mixed[i_th_drug_index]
                    k_th_drug = mixed[k_th_drug_index]
                    j_th_virus = mixed_2[j_th_virus_index]

                    tempX.append(np.concatenate((i_th_drug, k_th_drug , j_th_virus )))


                #print('shape of tempx:' , np.shape(tempX))

                tempX = tf.expand_dims(tempX, -1)

                res = model.predict(
                    tempX
                )
                    
                for iTemp in range(len(res)):
                    #print(res[iTemp])
                    testRes.append(res[iTemp][0])
            #print(testRes)
            
            prediction_treshhold = threshold_tuning(testY, testRes, mode='PR', step=0.01)
            #print(testRes)
            # print('len temp x: ' , len(tempX))
            #np.savetxt(directory2 + '/y_hat.txt', testRes)
            #print(len(testRes))
            #print(type(testRes))
            #print(testY,testRes)
            y_pred = []
            for i_pred in range(len(testRes)):
                if testRes[i_pred] > prediction_treshhold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            #print(y_pred)
                    
            
            evaluation_metrics = evaluate(testY, y_pred)

            nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(testY, y_pred)
            auc_keras = auc(nn_fpr_keras, nn_tpr_keras)

            precision, recall, thresholds = precision_recall_curve(testY, y_pred)
            aupr_keras = auc(recall, precision)
            
                
            
            auc_roc_test_list.append([outer_fold_no, epoch, auc_keras , aupr_keras])

            for i in range(len(evaluation_metrics)):
                inner_fold_metrcis[i].append(evaluation_metrics[i])
                outer_fold_metrics[i].append(evaluation_metrics[i])
                

            inner_fold_metrcis[6].append(auc_keras)
            inner_fold_metrcis[7].append(aupr_keras)
            outer_fold_metrics[6].append(auc_keras)
            outer_fold_metrics[7].append(aupr_keras)


            
            #auc_roc_test_list.append([epoch, auc_keras ])
            print('outer fold number: ', outer_fold_no, 'inner fold number: ', inner_fold_no, 'epoch: ' , epoch, 'auc: ', auc_keras, 'aupr: ', aupr_keras)
            
            np.savetxt('../'+result_path+'/DRaW_ratio'+str(ratio)+'_10fold.csv', auc_roc_test_list, delimiter=',', fmt='%s')
            
            #save best model up to here
            if aupr_keras > max_aupr:

                # display = RocCurveDisplay(fpr=nn_fpr_keras, tpr=nn_tpr_keras, roc_auc= auc_keras, estimator_name=' auc')
                # display.plot()
                # pyplot.show()
                
                np.savetxt(directory3 + '/y_hat_eval.txt', y_pred)
                
                pyplot.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='auc')
                #axis labels
                pyplot.xlabel('False Positive Rate')
                pyplot.ylabel('True Positive Rate')
                #show the legend
                pyplot.legend()
                #show the plot
                #pyplot.show()
                pyplot.savefig(directory3 + '/auc.png')
                pyplot.clf()

                pyplot.plot(recall, precision, marker='.', label='AUPR')
                #axis labels
                pyplot.xlabel('Recall')
                pyplot.ylabel('Precision')
                #show the legend
                pyplot.legend()
                #show the plot
                #pyplot.show()
                pyplot.savefig(directory3  +  '/aupr.png')
                pyplot.clf()

                log_file2 = open(directory3 + '/best_evaluation_log.txt', 'w')
                log_file2.write('Accuracy => {}'.format(evaluation_metrics[0]))
                log_file2.write('\nPrecision => {}'.format(evaluation_metrics[1]))
                log_file2.write('\nRecall => {}'.format(evaluation_metrics[2]))
                log_file2.write('\nSpecificity => {}'.format(evaluation_metrics[3]))
                log_file2.write('\nF1-score => {}'.format(evaluation_metrics[4]))
                log_file2.write('\nMCC => {}'.format(evaluation_metrics[5]))
                log_file2.write('\nAUROC => {}'.format(auc_keras))
                log_file2.write('\nAUPR => {}'.format(aupr_keras))
                log_file2.write('\nTreshhold => {}'.format(prediction_treshhold))
                log_file2.close()
                model.save('../'+result_path+'/DRaW_ratio'+str(ratio)+'_10fold'+str(outer_fold_no)+'.h5')
                #np.savetxt(fname='testRes_fold'+ str(fold_no) + '_ratio'+str(ratio)+'.txt' , X = testRes)
                #np.savetxt(fname='testY_fold'+ str(fold_no) + '_ratio'+str(ratio)+'.txt' , X = testY)
                max_aupr = aupr_keras
                best_model = model
        inner_fold_no += 1


        print('test:')

        testIT = len(X_testIndex) // batch_size
        #print(len(X_testIndex))
        testRes = []
        testY = []
        for it in range(testIT + 1):
            if it < testIT:
                indexRange = [it * batch_size, (it + 1) * batch_size]
            else:
                indexRange = [it * batch_size, len(X_testIndex)]
            #print(indexRange)
            tempX = []
            for cIT in range(indexRange[0], indexRange[1]):
                index = X_testIndex[cIT]
                testY.append(
                    Y_test[cIT]
                )
                # Concat DrugSim(index[0]) and ProteinSim(index[1]) => tempX
        
                i_th_drug_index = index[0]
                k_th_drug_index = index[1]
                j_th_virus_index = index[2]
                
            

                #i_th_drug= similarity_matrix_drug_disease[i_th_drug_index]
                i_th_drug= mixed[i_th_drug_index]
                k_th_drug = mixed[k_th_drug_index]
                j_th_virus = mixed_2[j_th_virus_index]

                tempX.append(np.concatenate((i_th_drug, k_th_drug , j_th_virus )))


            #print('shape of tempx:' , np.shape(tempX))

            tempX = tf.expand_dims(tempX, -1)

            res = best_model.predict(
                tempX
            )
                
            for iTemp in range(len(res)):
                #print(res[iTemp])
                testRes.append(res[iTemp][0])
        #print(testRes)
        np.savetxt(directory3 + '/y_hat_not_rounded_test.txt', testRes)
        prediction_treshhold = threshold_tuning(testY, testRes, mode='PR', step=0.01)
        #print(testRes)
        # print('len temp x: ' , len(tempX))
        #print(len(testRes))
        #print(type(testRes))
        #print(testY,testRes)
        y_pred = []
        for i_pred in range(len(testRes)):
            if testRes[i_pred] > prediction_treshhold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #print(y_pred)
        np.savetxt(directory3 + '/y_hat_test.txt', y_pred)
        evaluation_metrics = evaluate(testY, y_pred)

        nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(testY, testRes)
        auc_keras = auc(nn_fpr_keras, nn_tpr_keras)

        precision, recall, thresholds = precision_recall_curve(testY, testRes)
        aupr_keras = auc(recall, precision)
        
        
        pyplot.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='auc')
        #axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        #show the legend
        pyplot.legend()
        #show the plot
        #pyplot.show()
        pyplot.savefig(directory3 + '/Test-auc.png')
        pyplot.clf()
        pyplot.plot(recall, precision, marker='.', label='AUPR')
        #axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        #show the legend
        pyplot.legend()
        #show the plot
        #pyplot.show()
        pyplot.savefig(directory3 +  '/Test-aupr.png')
        pyplot.clf()
            
        
        auc_roc_test_list.append([outer_fold_no, epoch, auc_keras , aupr_keras])
        
        #auc_roc_test_list.append([epoch, auc_keras ])
        #print('fold number:', fold_no, 'epoch: ' , epoch, 'auc: ', auc_keras, 'aupr: ', aupr_keras)
        
        #np.savetxt(result_path+'/DEDTI_ratio'+str(ratio)+'_10fold.csv', auc_roc_test_list, delimiter=',', fmt='%s')
        
        
        for i in range(len(evaluation_metrics)):
           test_metrics[i].append(evaluation_metrics[i])

        test_metrics[6].append(auc_keras)
        test_metrics[7].append(aupr_keras)

        #save best model up to here
        log_file2 = open(directory3 + '/log test.txt', 'w')
        log_file2.write('Accuracy => {}'.format(evaluation_metrics[0]))
        log_file2.write('\nPrecision => {}'.format(evaluation_metrics[1]))
        log_file2.write('\nRecall => {}'.format(evaluation_metrics[2]))
        log_file2.write('\nSpecificity => {}'.format(evaluation_metrics[3]))
        log_file2.write('\nF1-score => {}'.format(evaluation_metrics[4]))
        log_file2.write('\nMCC => {}'.format(evaluation_metrics[5]))
        log_file2.write('\nAUROC => {}'.format(auc_keras))
        log_file2.write('\nAUPR => {}'.format(aupr_keras))
        log_file2.write('\nTreshhold => {}'.format(prediction_treshhold))
        log_file2.close()
        #model.save(result_path+'/DEDTI_ratio'+str(ratio)+'_10fold'+str(fold_no)+'.h5')
        #np.savetxt(fname='testRes_fold'+ str(fold_no) + '_ratio'+str(ratio)+'.txt' , X = testRes)
        #np.savetxt(fname='testY_fold'+ str(fold_no) + '_ratio'+str(ratio)+'.txt' , X = testY)
        #max_aupr = aupr_keras
        print('max of aupr in fold ' + str(outer_fold_no) + ' is:', max_aupr)
    log_file = open(directory2 + '/' + str(inner_fold_no) +' inner_fold_log.txt', 'w')          
    log_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[0]), np.std(inner_fold_metrcis[0])))
    log_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[1]), np.std(inner_fold_metrcis[1])))
    log_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[2]), np.std(inner_fold_metrcis[2])))
    log_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[3]), np.std(inner_fold_metrcis[3])))
    log_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[4]), np.std(inner_fold_metrcis[4])))
    log_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[5]), np.std(inner_fold_metrcis[5])))
    log_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[6]), np.std(inner_fold_metrcis[6])))
    log_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(inner_fold_metrcis[7]), np.std(inner_fold_metrcis[7])))
    log_file.close()
    outer_fold_no = outer_fold_no + 1
  # save test metrics
  log_file = open(directory + '/average_test_fold_log.txt', 'w')          
  log_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(test_metrics[0]), np.std(test_metrics[0])))
  log_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(test_metrics[1]), np.std(test_metrics[1])))
  log_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(test_metrics[2]), np.std(test_metrics[2])))
  log_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(test_metrics[3]), np.std(test_metrics[3])))
  log_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(test_metrics[4]), np.std(test_metrics[4])))
  log_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(test_metrics[5]), np.std(test_metrics[5])))
  log_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(test_metrics[6]), np.std(test_metrics[6])))
  log_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(test_metrics[7]), np.std(test_metrics[7])))
  log_file.close()

  # save evaluate metrics
  log_file = open(directory + '/average_all_folds_evaluation_log.txt', 'w')          
  log_file.write('\nAccuracy => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[0]), np.std(outer_fold_metrics[0])))
  log_file.write('\nPrecision => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[1]), np.std(outer_fold_metrics[1])))
  log_file.write('\nRecall => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[2]), np.std(outer_fold_metrics[2])))
  log_file.write('\nSpecificity => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[3]), np.std(outer_fold_metrics[3])))
  log_file.write('\nF1-score => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[4]), np.std(outer_fold_metrics[4])))
  log_file.write('\nMCC => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[5]), np.std(outer_fold_metrics[5])))
  log_file.write('\nAUROC => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[6]), np.std(outer_fold_metrics[6])))
  log_file.write('\nAUPR => AVG : {}  STDEV : {}'.format(np.average(outer_fold_metrics[7]), np.std(outer_fold_metrics[7])))
  log_file.close()


    
   
    

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser() #description="training data")
  parser.add_argument('--data_path', type=str, required=True)
  parser.add_argument('--ratio', type=int, required= True)
  parser.add_argument('--result_path', type=str, required= True)

  args = parser.parse_args()
  config = vars(args)
  print(config)

  train_DRaW_model(args.data_path, args.ratio, args.result_path)
