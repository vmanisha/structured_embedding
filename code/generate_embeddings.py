import argparse as ap
import numpy as np 
import pandas as pd
import scipy
import lxml.html as lh
from nltk.tokenize import word_tokenize
from collections import deque
keep_tags = set(['h1','h2','h3','h4','b','i','strong',\
                    'h5','h6','table','ul','ol','img', 'a'])

def encodeUTF(string):
  newString = ''
  for ch in string:
    try:
      nch = ch.encode('ascii', 'ignore')
    except UnicodeDecodeError, error:
      #print error.args, error.start, error.end, ch
      nch = ''
    newString += nch
  return newString



def LoadEmbeddings(glove_file):
    embedding_index = {}
    words = 0
    
    for line in open(glove_file, 'r'):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
        words+=1
        if words == 1000000:
            print words
    print 'Loaded word embeddings for ', words 
    return embedding_index


def GenerateDocumentEmbedding(document_text, glove_vectors):
    node_embedding = {}
    vector_count_per_tag = {}
    oovocabulary_words = 0
    document_text = document_text.lower()
    page_object = lh.fromstring(document_text)
    total_tags = 0
    #remove comment
    # Links, headings, tables, images, bold, italics, list
    for tag in keep_tags:
        node_embedding[tag] = np.zeros(300, dtype=np.float)
        vector_count_per_tag[tag]=0.0
    
    for ele in page_object.iter():
        total_tags +=1
        if ele.tag in keep_tags:
            content = []
            if ele.text:
                content+=word_tokenize(ele.text)
            if ele.tag == 'img' and 'alt' in ele.attrib:
                content+=word_tokenize(ele.attrib['alt']) 
            if ele.tag == 'img' and 'title' in ele.attrib:
                content+=word_tokenize(ele.attrib['title']) 
            if ele.tag == 'a' and 'href' in ele.attrib:
                content+=word_tokenize(ele.attrib['href']) 
            
            all_child_nodes = deque([])
            for child in ele.iterchildren():
                all_child_nodes.append(child)
            
            while len(all_child_nodes) > 0:
                curr_node = all_child_nodes.popleft()
                if curr_node.text:
                    content+=word_tokenize(curr_node.text)
                for child in curr_node.iterchildren():
                    all_child_nodes.append(child)
            
            # word list
            for word in content:
                if word in glove_vectors:
                    node_embedding[ele.tag]+=glove_vectors[word]
                    vector_count_per_tag[ele.tag]+=1.0
                else:
                    oovocabulary_words+=1
    
    for tag in node_embedding.keys():
        if vector_count_per_tag[tag] > 0:
            node_embedding[tag]/=vector_count_per_tag[tag]

    return total_tags, oovocabulary_words, vector_count_per_tag, node_embedding



def GenerateQueryEmbedding(query_text, glove_vectors):
    query_embedding = np.zeros(300, dtype=np.float)
    oov_words = 0
    word_count = 0
    for word in query_text.split():
        word_count+=1.0
        if word in glove_vectors:
            query_embedding+=glove_vectors[word]
        else:
            oov_words+=1
        
    if word_count-oov_words > 0:
        query_embedding/=(1.0 * (word_count-oov_words))

    return oov_words, query_embedding
    


def main():
    parser = ap.ArgumentParser(description='Generate word embeddings from trec documents.')
    parser.add_argument('-t','--trecFile',required=True, help='Trec file containing all documents.')
    parser.add_argument('-g','--gloveFolder', required=True, help='Folder with glove embeddings')
    parser.add_argument('-o','--outFile', required=True, help='File to output embeddings.')

    args = parser.parse_args()
    out_file = open(args.outFile,'w')

    glove_word_embedding = LoadEmbeddings(args.gloveFolder)
    doc_count=0
    for line in open(args.trecFile, 'r'):
        doc_count+=1

        split = line.split('\t')
        doc_text = split[7]
        query_text = split[2]
        oov_words_query, query_embedding = GenerateQueryEmbedding(query_text, glove_word_embedding)
        total_tags, oov_words, words_per_tag, document_embedding = GenerateDocumentEmbedding(encodeUTF(doc_text), glove_word_embedding)
        # write to file              	
        out_file.write('\t'.join(split[0:6])+'\t'+str(oov_words_query)+'\t'+str(oov_words))
        out_file.write('\t'+' '.join(['{:.2f}'.format(i) for i in query_embedding]))
        out_file.write('\t'+str(total_tags)) 
        for tag in keep_tags:
            if tag in document_embedding:
                out_file.write('\t'+tag+':::'+str(words_per_tag[tag])+':::'+' '.join(['{:.2f}'.format(i) for i in document_embedding[tag]]))
        out_file.write('\n')
        if doc_count % 10000 == 0:
            print doc_count
    out_file.close()
    


if __name__ == '__main__':
    main()
