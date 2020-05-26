#python scrpit/prep.py

#python tools/learn_bpe.py --symbols 35000 --input data/en-train.txt --output data/en.bpe
#python tools/learn_bpe.py --symbols 35000 --input data/zh-train.txt --output data/zh.bpe
echo "finish bpe"

for TYPE in en zh;
do
	for DATA in dev val train;
	do
		echo apply bpe on data/$TYPE-$DATA.txt
		python tools/apply_bpe.py --code data/$TYPE.bpe --input data/$TYPE-$DATA.txt --output data/$TYPE-$DATA-bpe.txt 
		python scrpit/process_tag.py data/$TYPE-$DATA-bpe.txt 
	done
done


python preprocess.py \
	-num_threads 20 \
	-train_src data/zh-train-bpe.txt -train_tgt data/en-train-bpe.txt \
	-valid_src data/zh-val-bpe.txt -valid_tgt data/en-val-bpe.txt \
	-save_data data/zh2en -overwrite \
	-src_seq_length 200 -tgt_seq_length 200
