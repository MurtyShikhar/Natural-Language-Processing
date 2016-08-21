python features_submit.py $1 tmp/inp_features.txt 0
java -cp "mallet/class:mallet/lib/mallet-deps.jar"  cc.mallet.fst.SimpleTagger --model-file best_model tmp/inp_features.txt > tmp/labels.txt
python submit.py $1 tmp/labels.txt $2



