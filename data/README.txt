Microsoft Research WikiQA Code Package

-------------------------------------------------------------------------------------------------------
Contact Persons: Yi Yang                  Georgia Tech (yangyiycc@gmail.com)
				 Scott Wen-tau Yih        Microsoft Research (scottyih@microsoft.com)
-------------------------------------------------------------------------------------------------------

We release the code of some models used in our EMNLP-2015 paper: WikiQA: A Challenge Dataset for Open-Domain Question Answering.

Please cite it if you use this code.

Yi Yang, Wen-tau Yih and Christopher Meek. "WikiQA: A Challenge Dataset for Open-Domain Question Answering." In EMNLP-2015.

@InProceedings{YangYihMeek:EMNLP2015:WikiQA,
  author    = {Yang, Yi  and  Yih, Wen-tau  and  Meek, Christopher},
  title     = {{WikiQA}: {A} Challenge Dataset for Open-Domain Question Answering},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {September},
  year      = {2015},
  address   = {Lisbon, Portugal},
  publisher = {Association for Computational Linguistics}
}


-------------------------------------------------------------------------------------------------------


Version 1.0: October 30, 2015

-------------------------------------------------------------------------------------------------------

The code includes implementation of two models: Convolutional Neural Networks (CNN), and Logistic regression with CNN and count features (CNN-Cnt).

You can simply reproduce results using "./go_trec.sh" (for QASent dataset) and "./go_wiki.sh" (for WikiQA dataset).

Some things required:
    1. We are not able to release QASent dataset, and you need to prepare the needed files and put them in the "./data/trec" folder. Please refer to the files in the "./data/wiki" folder.
	2. You need to download pre-trained word embeddings (https://code.google.com/p/word2vec/) to the "./data" folder.
	3. You need to compile the official TREC evaluation software for your operating system and copy the runnable binary (./trec_eval) to the "." folder

You can tune some parameters by looking into the "tuning_cnn" method in "qa_score.py".
