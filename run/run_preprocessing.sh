#!/bin/bash

# local

# db_path='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/database'
# train_data='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/train_vitext2sql.json'
# dev_data='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/dev.json'
# table_data='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/tables.json'
# train_out='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/linegraph_out/train.lgesql.bin'
# dev_out='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/linegraph_out/dev.lgesql.bin'
# table_out='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/linegraph_out/tables.bin'
# vocab_phow2v='D:/GitHub-Work-Space/text2sql-lgesql/third_party/phow2v_emb/word2vec_vi_syllables_300dims.txt'
# vocab_out='D:/GitHub-Work-Space/text2sql-lgesql/data/vitext2sql_syllable_level/linegraph_out/vocab_pretrained.txt'

# echo "Start to preprocess the original train dataset ..."
# python -u D:\\GitHub-Work-Space\\text2sql-lgesql\\preprocess\\process_dataset.py --db_dir ${db_path} --dataset_path ${train_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path ${train_out} #--verbose > train.log
# echo "Start to preprocess the original dev dataset ..."
# python -u D:\\GitHub-Work-Space\\text2sql-lgesql\\preprocess\\process_dataset.py --dataset_path ${dev_data} --table_path ${table_out} --output_path ${dev_out} #--verbose > dev.log
# echo "Start to build word vocab for the dataset ..."
# python -u D:\\GitHub-Work-Space\\text2sql-lgesql\\preprocess\\build_phow2v_vocab.py --data_paths ${train_out} --table_path ${table_out} --reference_file ${vocab_phow2v} --mwf 4 --output_path ${vocab_out}
# echo "Start to construct graphs for the dataset ..."
# python -u D:\\GitHub-Work-Space\\text2sql-lgesql\\preprocess\\process_graphs.py --dataset_path ${train_out} --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
# python -u D:\\GitHub-Work-Space\\text2sql-lgesql\\preprocess\\process_graphs.py --dataset_path ${dev_out} --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}

#!/bin/bash

# colab 

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
python3 -u /content/text2sql-lgesql/preprocess/process_graphs.py --dataset_path ${train_out} --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
python3 -u /content/text2sql-lgesql/preprocess/process_graphs.py --dataset_path ${dev_out} --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}
