#!/bin/bash

# * Author: Kaixin Wu               * #
# * Email : wukaxin_neu@163.com     * #
# * Date  : 12/38/2017              * #
# * Time  : 11:19                   * #
# * evaluate bleu score.            * #

decode_file=$1

## Generate XML file
perl NiuTrans-generate-xml-for-mteval.pl \
    -1f $decode_file \
    -tf dev.txt \
    -rnum 4

## Evaluate bleu score
perl mteval-v13a.pl \
     -r ref.xml \
     -s src.xml \
     -t tst.xml

## Remove temp file
rm ref.xml src.xml tst.xml
rm ${decode_file}.temp
