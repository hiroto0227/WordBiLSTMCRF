.PHONY: preprocess evaluate

train_define:
	echo "Char Level Redundant + 4000\n"

parse-large_corpus:
	#python3.7 large_corpus/parse_PubMed.py --input-dir ./Repository/LargeCorpus/PubMed --output-path ./Repository/LargeCorpus/Corpus.txt

parse-medline:
	#python3.7 large_corpus/parse_medline.py --input-dir ./Repository/LargeCorpus/medline --output-path ./Repository/LargeCorpus/parsed_corpus.txt

train-sentencepiece:
	python3.7 sentencepieces/sp_train.py --vocab-size 64000 --input-path ./Repository/LargeCorpus/parsed_corpus.txt;
	python3.7 sentencepieces/sp_train.py --vocab-size 32000 --input-path ./Repository/LargeCorpus/parsed_corpus.txt;
	python3.7 sentencepieces/sp_train.py --vocab-size 16000 --input-path ./Repository/LargeCorpus/parsed_corpus.txt;
	python3.7 sentencepieces/sp_train.py --vocab-size 8000 --input-path ./Repository/LargeCorpus/parsed_corpus.txt;
	#python3.7 sentencepieces/sp_train.py --vocab-size 4000 --input-path ./Repository/LargeCorpus/parsed_corpus.txt;
	python3.7 sentencepieces/sp_train.py --vocab-size 2000 --input-path ./Repository/LargeCorpus/parsed_corpus.txt;
	mv ./sp*.model ./Repository/SentencePieceModel/;
	mv ./sp*.vocab ./Repository/SentencePieceModel/

pretraining-dataload:
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_regex.txt;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp64000.txt --sp-model ./Repository/SentencePieceModel/sp64000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp32000.txt --sp-model ./Repository/SentencePieceModel/sp32000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp16000.txt --sp-model ./Repository/SentencePieceModel/sp16000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp8000.txt --sp-model ./Repository/SentencePieceModel/sp8000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp4000.txt --sp-model ./Repository/SentencePieceModel/sp4000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/parsed_corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp2000.txt --sp-model ./Repository/SentencePieceModel/sp2000.model;

# pretraining-fasttext:
# 	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp64000.txt -output ./Repository/Pretrained/ft_pretrain_sp64000.model -dim 50  -epoch 20;
# 	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp32000.txt -output ./Repository/Pretrained/ft_pretrain_sp32000.model -dim 50  -epoch 20;
# 	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp16000.txt -output ./Repository/Pretrained/ft_pretrain_sp16000.model -dim 50  -epoch 20;
# 	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp8000.txt -output ./Repository/Pretrained/ft_pretrain_sp8000.model -dim 50  -epoch 20;
# 	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp4000.txt -output ./Repository/Pretrained/ft_pretrain_sp4000.model -dim 50  -epoch 20;
# 	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp2000.txt -output ./Repository/Pretrained/ft_pretrain_sp2000.model -dim 50  -epoch 20;

pretraining-glove:
	#python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_regex.txt --output-path ./Repository/Pretrained/gv_pretrain_regex.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp64000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp64000.model --vector-size 50
	#python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp32000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp32000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp16000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp16000.model --vector-size 50
	#python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp8000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp8000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp4000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp4000.model --vector-size 50
	#python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp2000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp2000.model --vector-size 50

preprocess:
	python3.7 preprocess/preprocess.py --input-dir ./Repository/Chemdner/train --output-path ./Repository/SeqData/train.csv;
	python3.7 preprocess/preprocess.py --input-dir ./Repository/Chemdner/valid --output-path ./Repository/SeqData/valid.csv;
	python3.7 preprocess/preprocess.py --input-dir ./Repository/Chemdner/test --output-path ./Repository/SeqData/test.csv;

train:
	python3.7 model/wordRedundant.py --mode train --config-path model/baseline.config;

predict:
	python3.7 model/wordRedundant.py --mode predict --config-path model/baseline.config;

evaluate:
	python3.7 evaluate/evaluate.py --schema bioes --input-path ./Repository/SeqData/predicted.csv --verbose-num 100;
