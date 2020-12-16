cd code

# create trec datasets and extract count features
python -u process_data.py --w2v_fname ../data/GoogleNews-vectors-negative300.bin --extract_feat 1 ../data/wiki/WikiQASent-train.txt ../data/wiki/WikiQASent-dev.txt ../data/wiki/WikiQASent-test.txt ../wiki_cnn.pkl

# ----- Convolutional Neural Networks -----
python -u qa_score.py --dev_refname ../data/wiki/WikiQASent-dev.ref  --test_refname  ../data/wiki/WikiQASent-test.ref --dev_ofname ../pred/wiki/cnn-dev.rank --test_ofname ../pred/wiki/cnn-test.rank --cnn_cnt 0 ../wiki_cnn.pkl

../trec_eval -c ../data/wiki/WikiQASent-dev-filtered.ref ../pred/wiki/cnn-dev.rank
../trec_eval -c ../data/wiki/WikiQASent-test-filtered.ref ../pred/wiki/cnn-test.rank

# ----- Convolutional Neural Networks + Count features -----
python -u qa_score.py --dev_refname ../data/wiki/WikiQASent-dev.ref  --test_refname  ../data/wiki/WikiQASent-test.ref --dev_ofname ../pred/wiki/cnn-cnt-dev.rank --test_ofname ../pred/wiki/cnn-cnt-test.rank --cnn_cnt 1 ../wiki_cnn.pkl

../trec_eval -c ../data/wiki/WikiQASent-dev-filtered.ref ../pred/wiki/cnn-cnt-dev.rank
../trec_eval -c ../data/wiki/WikiQASent-test-filtered.ref ../pred/wiki/cnn-cnt-test.rank

cd ..



