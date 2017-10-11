import argparse as ap
from keras.layers import Input, Conv1D, Conv2D, Flatten, Dropout, Dense
from keras.models import Sequential
from keras.layers.pooling import MaxPooling1D,MaxPooling2D
from keras.layers.core import Dense
from keras.models import Model
from keras import metrics
from keras.layers.merge import Multiply
from evaluation.metrics import ComputeNDCGWithList
import numpy as np
import random
import pandas as pd
from numpy import linalg as LA

def InitializeModel():
    query_input = Input(shape=(300,1), dtype='float32', name='query_input')
    query_conv = Conv1D(filters=50, kernel_size=3, strides=1, padding='same', activation='relu')(query_input)
    query_pool = MaxPooling1D(pool_size=2)(query_conv)
    #query_conv1 = Conv1D(filters=50, kernel_size=3, strides=1, padding='same', activation='relu')(query_pool)
    #query_pool1 = MaxPooling1D(pool_size=2)(query_conv1)
    query_flat = Flatten()(query_pool)
    query_dense = Dense(units = 25, activation='relu')(query_flat)

    doc_input =Input(shape=(300,1), dtype='float32', name='doc_input')
    doc_conv = Conv1D(filters=50, kernel_size=3, strides=1, padding='same', activation='relu')(doc_input)
    doc_pool = MaxPooling1D(pool_size=2)(doc_conv)
    doc_flat = Flatten()(doc_pool)
    doc_dense = Dense(units = 25, activation='relu')(doc_flat)
    
    query_doc_vector = Multiply()([query_dense, doc_dense])
    query_doc_dense = Dense(units = 5, activation='softmax', name='relevance')(query_doc_vector)

    model= Model(inputs=[query_input, doc_input], outputs=query_doc_dense)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])

    print model.summary()
    return model



def TrainAndEvaluateModel(model, train_query_features, train_doc_features, train_relevance,\
        test_query_features, test_doc_features, test_relevance):
    model.fit({'query_input': train_query_features, 'doc_input':train_doc_features},\
            {'relevance':train_relevance}, epochs=50, batch_size=100,\
            validation_split=0.1, verbose=2, class_weight = {0:1, 1:10, 2:20, 3:30, 4:50})

    loss, mae, acc = model.evaluate({'query_input': test_query_features, 'doc_input':test_doc_features},\
            {'relevance':test_relevance}, batch_size=100)

    test_predictions = model.predict({'query_input': test_query_features, 'doc_input':test_doc_features}, batch_size=100)
    
    return loss, mae, acc, test_predictions


def GetRelVector(rel_val):
    rel_vector = np.zeros(5)
    if rel_val <= 0:
        rel_vector[0]=1
        return rel_vector
    elif rel_val == 1:
        rel_vector[1]=1
        return rel_vector
    elif rel_val == 2:
        rel_vector[2]=1
        return rel_vector
    elif rel_val == 3:
        rel_vector[3]=1
        return rel_vector
    elif rel_val == 4:
        rel_vector[4]=1
        return rel_vector

def main():
    parser = ap.ArgumentParser(description='Train model to predict document relevance')
    parser.add_argument('-v','--vectorsFile', required = True, help='File containing query, document vectors and relevance info')
    parser.add_argument('-o','--outResultFile', required = True, help='File to output ranking results')
    parser.add_argument('-b','--beginTestId', required = True, help='Beginning query id to test')
    parser.add_argument('-e','--endTestId', required = True, help='End query id to test')

    args = parser.parse_args()
    all_rows = []
    ind = 0
    for line in open(args.vectorsFile,'r'):
        split= line.split('\t')
        # Read query vector and document vector
        query_id = int(split[0])
        doc_id = split[4]
        query_vector =np.asarray(split[8].split(), dtype='float32')
        if np.sum(query_vector) == 0:
            print 'Query vector null', query_id
        # Read the relevance label
        rel_num = int(split[5])
        rel_vector = GetRelVector(rel_num)
        
        all_vectors = {}
        for entry in split[10:]:
            tag_split=entry.split(':::')
            tag_name = tag_split[0]
            tag_words = tag_split[1]
            tag_vector = np.asarray(tag_split[2].split(), dtype='float32')
	    if tag_name in ['h1','h2','h3','h4','h5','h6']:
            	if 'head' not in all_vectors:
		    all_vectors['head'] = np.zeros(300, dtype='float32')
		all_vectors['head'] += tag_vector
	    elif tag_name in ['b','strong','i']:
            	if 'bold_italics' not in all_vectors:
		    all_vectors['bold_italics'] = np.zeros(300, dtype='float32')
		all_vectors['bold_italics'] += tag_vector
	    elif tag_name in ['ul','li']:
            	if 'list' not in all_vectors:
		    all_vectors['list'] = np.zeros(300, dtype='float32')
		all_vectors['list'] += tag_vector
	    else:
		all_vectors[tag_name] = tag_vector
	all_vectors['head']/=6
	all_vectors['bold_italics']/=2
	all_vectors['list']/=2

        all_rows.append({'query_id':query_id,'doc_id':doc_id, 'qv':query_vector,\
                'dv':all_vectors, 'rel_num': rel_num, 'rel':rel_vector})
        #if ind == 1000:
        #    break
        ind+=1

    for tag in ['head','table','a','list','img']:
        train_query_features = []
        train_doc_features = []
        train_rel = []
        
        test_query_features = []
        test_doc_features = []
        test_rel = []

        train_count = 0
        test_count = 0

        start_query_id = int(args.beginTestId)
        end_query_id = int(args.endTestId)

        test_query_id_doc_pos = {}
        test_query_id_doc_rel = {}
        
        for entry in all_rows:
            if entry['query_id'] >= start_query_id and entry['query_id'] < end_query_id:
                if entry['query_id'] not in test_query_id_doc_pos:
                    test_query_id_doc_pos[entry['query_id']] = []
                    test_query_id_doc_rel[entry['query_id']] = []
                
                test_query_features.append(entry['qv'])
                test_doc_features.append(entry['dv'][tag])
                test_rel.append(entry['rel'])
                test_query_id_doc_pos[entry['query_id']].append(test_count)
                test_query_id_doc_rel[entry['query_id']].append(entry['rel_num'])
                
                test_count+=1
            else:
                to_add = True
                if entry['rel_num'] <=0 and random.random() < 0.7:
                    to_add = False

                if to_add:
                    train_query_features.append(entry['qv'])
                    train_doc_features.append(entry['dv'][tag])
                    train_rel.append(entry['rel'])
                    train_count+=1

        train_query_features = np.reshape(np.asarray(train_query_features, dtype='float32'), (train_count,300,1))
        train_doc_features = np.reshape(np.asarray(train_doc_features, dtype='float32'), (train_count,300,1))
        train_rel = np.asarray(train_rel, dtype='int32')
        
        test_query_features = np.reshape(np.asarray(test_query_features, dtype='float32'), (test_count,300,1))
        test_doc_features = np.reshape(np.asarray(test_doc_features, dtype='float32'), (test_count,300,1))
        test_rel = np.asarray(test_rel, dtype='int32')
        
        print tag, test_rel.shape, train_query_features.shape, test_doc_features.shape
        
        model = InitializeModel()   
        
        loss, mae, accuracy, test_predictions = TrainAndEvaluateModel(model,train_query_features , train_doc_features, train_rel,\
            test_query_features, test_doc_features, test_rel)

        print tag, 'Test loss, mae, accuracy', loss, mae, accuracy
        
        test_ndcg10  = {}
        test_ndcg1  = {}
        for query_id, rel_list in test_query_id_doc_rel.items():
            predict_rel = []
            for pos in test_query_id_doc_pos[query_id]:
                predict_rel.append(np.argmax(test_predictions[pos]))
            test_ndcg10[query_id] = ComputeNDCGWithList(predict_rel, rel_list, 10)
            test_ndcg1[query_id]  = ComputeNDCGWithList(predict_rel, rel_list, 1)
        
        ndcg_table = pd.DataFrame.from_dict(test_ndcg1, orient='index')
        ndcg_table.columns=['ndcg']
        ndcg_table.to_csv(args.outResultFile+'_'+tag+'_ndcg1_'+str(start_query_id)+'_'+str(end_query_id), sep='\t')
        ndcg_table = pd.DataFrame.from_dict(test_ndcg10, orient='index')
        ndcg_table.columns=['ndcg']
        ndcg_table.to_csv(args.outResultFile+'_'+tag+'_ndcg10_'+str(start_query_id)+'_'+str(end_query_id), sep='\t')

        print tag, 'NDCG@1 on test queries', np.mean(test_ndcg1.values()), np.std(test_ndcg1.values())
        print tag, 'NDCG@10 on test queries', np.mean(test_ndcg10.values()), np.std(test_ndcg10.values())


if __name__ == '__main__':
    main()        


