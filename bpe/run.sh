#!/bin/bash

# *   Function        : Byte Pair Encoding Segment 
# *   Author          : Wu Kaixin
# *   Date            : 2018/1/5
# *   Email           : wukaixin_neu@163.com
# *   Last Modified in: NEU NLP Lab., shenyang

input_file=$1
num_megrges=$2
tmp_dir=${input_file}.tmp
codes_file=${input_file}.codes_file
out_file=${input_file}.bpe
vocab_file=${out_file}.vocab

#setp1: Learn BPE
python learn_bpe.py -s $num_megrges < $input_file > $codes_file

#step2: BPE Segment
python apply_bpe.py -c $codes_file < $input_file > $out_file

#step3: Get Vocab
python get_vocab.py < $out_file > $vocab_file

if [[ ! -d "$tmp_dir" ]]; then  
　　mkdir "$tmp_dir"  
fi 

#mkdir $tmp_dir
mv $codes_file $tmp_dir
mv $out_file $tmp_dir
mv $vocab_file $tmp_dir  
