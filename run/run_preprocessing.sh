#!/bin/bash

# train_data='data/train.json'
# dev_data='data/dev.json'
# table_data='data/tables.json'
# train_out='data/train.lgesql.bin'
# dev_out='data/dev.lgesql.bin'
# table_out='data/tables.bin'
# vocab_glove='pretrained_models/glove.42b.300d/vocab_glove.txt'
# vocab='pretrained_models/glove.42b.300d/vocab.txt'

# echo "Start to preprocess the original train dataset ..."
# python3 -u preprocess/process_dataset.py --dataset_path ${train_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path 'data/train.bin' --skip_large #--verbose > train.log
# echo "Start to preprocess the original dev dataset ..."
# python3 -u preprocess/process_dataset.py --dataset_path ${dev_data} --table_path ${table_out} --output_path 'data/dev.bin' #--verbose > dev.log
# echo "Start to build word vocab for the dataset ..."
# python3 -u preprocess/build_glove_vocab.py --data_paths 'data/train.bin' --table_path ${table_out} --reference_file ${vocab_glove} --mwf 4 --output_path ${vocab}
# echo "Start to construct graphs for the dataset ..."
# python3 -u preprocess/process_graphs.py --dataset_path 'data/train.bin' --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
# python3 -u preprocess/process_graphs.py --dataset_path 'data/dev.bin' --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}

#!/bin/bash

db_path='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/database'
train_data='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/train_vitext2sql.json'
dev_data='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/dev.json'
table_data='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/tables.json'
train_out='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/linegraph_out/train.lgesql.bin'
dev_out='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/linegraph_out/dev.lgesql.bin'
table_out='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/linegraph_out/tables.bin'
vocab_phow2v='/content/text2sql-lgesql/third_party/phow2v_emb/word2vec_vi_words_300dims.txt'
vocab_out='/content/drive/MyDrive/Datasets/ratsql/datasets/vitext2sql_syllable_level/linegraph_out/vocab_pretrained.txt'

echo "Start to preprocess the original train dataset ..."
python3 -u /content/text2sql-lgesql/preprocess/process_dataset.py --db_dir ${db_path} --dataset_path ${train_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path ${train_out} #--verbose > train.log
echo "Start to preprocess the original dev dataset ..."
python3 -u /content/text2sql-lgesql/preprocess/process_dataset.py --dataset_path ${dev_data} --table_path ${table_out} --output_path ${dev_out} #--verbose > dev.log
echo "Start to build word vocab for the dataset ..."
python3 -u /content/text2sql-lgesql/preprocess/build_phow2v_vocab.py --data_paths ${train_out} --table_path ${table_out} --reference_file ${vocab_phow2v} --mwf 4 --output_path ${vocab_out}
echo "Start to construct graphs for the dataset ..."
python3 -u preprocess/process_graphs.py --dataset_path ${train_out} --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
python3 -u preprocess/process_graphs.py --dataset_path ${dev_out} --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}
