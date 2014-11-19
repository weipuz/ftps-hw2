#!/bin/bash
## shell script for filtering phrase-tables for small and medium corpora from the large phrase table extracted by moses (sh filterPP.sh medium)

root=/cs/natlang-projects/users/maryam/CMPT413/project
cd $root

SELL=bash

mert_dir=/cs/natlang-sw/Linux-x86_64/NL/MT/MOSES/GIT_20120619/bin   # Path of the MERT script
mert_scr=/cs/natlang-sw/Linux-x86_64/NL/MT/MOSES/GIT_20120619/scripts
root_dir=/cs/natlang-sw/Linux-x86_64/NL/MT/MOSES/GIT_20120619/bin

export PATH=$PATH:$mert_scr/training


data_dir=$1
src=$data_dir/train.cn                                        # Source file


# Filter the phrase table for the input source corpus
/usr/bin/perl $mert_scr/training/filter-model-given-input.pl $data_dir/phrase-table scripts/moses.ini $src
wait

