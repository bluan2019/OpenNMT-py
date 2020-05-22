#python tools/learn_bpe.py --symbols 35000 --input data/src-train.txt --output data/src.bpe
#python tools/learn_bpe.py --symbols 35000 --input data/tgt-train.txt --output data/tgt.bpe
#
#echo "finish bpe"
#
#for TYPE in src tgt;
#do
#	for DATA in dev val train;
#	do
#		echo apply bpe on data/$TYPE-$DATA.txt
#		python tools/apply_bpe.py --code data/$TYPE.bpe --input data/$TYPE-$DATA.txt --output data/$TYPE-$DATA-bpe.txt
#	done
#done

python preprocess.py -train_src data/src-train-bpe.txt -train_tgt data/tgt-train-bpe.txt -valid_src data/src-val-bpe.txt -valid_tgt data/tgt-val.txt -save_data data/en2zh -overwrite \
	-src_seq_length 200 -tgt_seq_length 200
