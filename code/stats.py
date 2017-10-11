import argparse as ap
import pandas as pd
import numpy as np
import scipy.stats

def main():
    parser = ap.ArgumentParser(description='Generate stats from features.')
    parser.add_argument('-f','--featureFile',required=True, help='File with clueweb documents.')
    parser.add_argument('-o','--outFile',required=True, help='File to write stats.')

    args = parser.parse_args()

    feature_frame = pd.read_csv(args.featureFile,sep='\t')
    # Focus on the following fields
    feature_frame['all_head']=feature_frame['h1']+feature_frame['h2']+feature_frame['h3']+\
            feature_frame['h4']+feature_frame['h5']+feature_frame['h6']
    feature_frame['all_strong']=feature_frame['b']+feature_frame['strong']

    total_documents = feature_frame.shape[0]
    print 'total documents', total_documents
    print 'all_head',feature_frame[feature_frame['all_head'] > 0].shape[0]
    print 'table',feature_frame[feature_frame['table'] > 0].shape[0]
    print 'i',feature_frame[feature_frame['i'] > 0].shape[0]
    print 'bold',feature_frame[feature_frame['all_strong'] > 0].shape[0]
    print 'a',feature_frame[feature_frame['a'] > 0].shape[0]
    print 'li',feature_frame[feature_frame['li'] > 0].shape[0]
    print 'img',feature_frame[feature_frame['img'] > 0].shape[0]
    
    
    print '-------------------PERCENTAGES--------------------------------\n\n'
    print 'all_head',feature_frame[feature_frame['all_head'] > 0].shape[0]/(total_documents*1.0)
    print 'table',feature_frame[feature_frame['table'] > 0].shape[0]/(total_documents*1.0)
    print 'i',feature_frame[feature_frame['i'] > 0].shape[0]/(total_documents*1.0)
    print 'bold',feature_frame[feature_frame['all_strong'] > 0].shape[0]/(total_documents*1.0)
    print 'a',feature_frame[feature_frame['a'] > 0].shape[0]/(total_documents*1.0)
    print 'li',feature_frame[feature_frame['li'] > 0].shape[0]/(total_documents*1.0)
    print 'img',feature_frame[feature_frame['img'] > 0].shape[0]/(total_documents*1.0)


    for element1 in ['all_head','all_strong','a','li','img','i','table']:
        for element2 in ['all_head','all_strong','a','li','img','i','table']:
            print element1, element2, feature_frame[(feature_frame[element1] > 0.0) & (feature_frame[element2] > 0.0)].shape[0]/(total_documents*1.0)

    for element1 in ['all_head','all_strong','a','li','img','i','table']:
        print element1, scipy.stats.pearsonr(feature_frame[element1], feature_frame['doc_rel'])



if __name__ =='__main__':
    main()

