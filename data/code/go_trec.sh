cd code

# create trec datasets and extract count features
python -u process_data.py --w2v_fname ../data/GoogleNews-vectors-negative300.bin --extract_feat 1 ../data/trec/train.txt ../data/trec/dev-filtered.txt ../data/trec/test-filtered.txt ../trec_cnn.pkl

# ----- Convolutional Neural Networks -----
python -u qa_score.py --dev_refname ../data/trec/dev-filtered.ref  --test_refname  ../data/trec/test-filtered.ref --dev_ofname ../pred/trec/cnn-dev.rank --test_ofname ../pred/trec/cnn-test.rank --cnn_cnt 0 ../trec_cnn.pkl

../trec_eval -c ../data/trec/Dev-T40.judgment ../pred/trec/cnn-dev.rank
../trec_eval -c ../data/trec/Test-T40.judgment ../pred/trec/cnn-test.rank

# ----- Convolutional Neural Networks + Count features -----
python -u qa_score.py --dev_refname ../data/trec/dev-filtered.ref  --test_refname  ../data/trec/test-filtered.ref --dev_ofname ../pred/trec/cnn-cnt-dev.rank --test_ofname ../pred/trec/cnn-cnt-test.rank --cnn_cnt 1 ../trec_cnn.pkl

../trec_eval -c ../data/trec/Dev-T40.judgment ../pred/trec/cnn-cnt-dev.rank
../trec_eval -c ../data/trec/Test-T40.judgment ../pred/trec/cnn-cnt-test.rank

cd ..



