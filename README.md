# structured_embedding
Keras model to create structured embeddings.

Depends on effort_analysis.evaluation.metrics.py to compute NDCG.

1. Generate the embeddings using generate_embeddings.py.
	Inputs: 
	a) Tab seperated file containing the following informaton:
		1)Query id, 2) Query type (ambigouos, faceted or single) 3) Query
		4)Query description, 5) doc id 6) doc rel 7) doc link 8) doc text
	b) Glove or word2vec vector file.
	c) File to output embeddings

2. Train the ranker using either train_relevance_ranker.py or train_relevance_conv1d_ranker.py
	Inputs:
   	a) File containing query, document vectors and relevance info
   	b) File to output ranking results
   	c) Test query id (start index_
   	d) Test query id (end index)
		
