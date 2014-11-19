Natural Language Processing: Course Project
-------------------------------------------

## Data Files

The data files for the course project are for a Chinese-English translation task.

### Training data

The training data is taken from the following sources:

* [Hong Kong Parliament parallel corpus](https://catalog.ldc.upenn.edu/LDC2004T08) 
* [GALE Phase-1 Chinese newsgroup data](https://catalog.ldc.upenn.edu/LDC2009T15).

**Warning: do not redistribute this data. SFU has a license to use this data from the Linguistic Data Consortium (LDC) but we cannot take this data and give it to others.**

Training data comes in different sizes. The data files in each of the
large, medium, and small folders are:

* `train.cn`: segmented Chinese corpus
* `train.cn.unseg`: un-segmented Chinese corpus
* `train.en`: lower-cased English corpus
* `phrase-table/moses/phrase-table.gz`: phrase-table in the usual format
  compatible with the [Moses SMT system](http://statmt.org/moses/)

#### Toy

First 2k sentences from the full training data.

#### Small

First 20k sentences from the full training data.

#### Medium

First 100k sentences from the full training data.

#### Large:

The entire training data (2.3M sentences).

In the `large` directory, there are a few additional files:

* `phrase-table/dev-filtered/rules_cnt.final.out`: phrase table
  filtered for dev, so that only the phrases useful for dev are in this
  phrase table.
* `phrase-table/test-filtered/rules_cnt.final.out`: phrase table filtered
  for test
* `lex.e2f` and `lex.f2e`: lexical probabilities 

### Tuning set

The files for tuning your SMT system are in the `dev` directory. This data
is meant to be used for tuning the weights of your machine translation
log-linear model. There are four references for each source sentence.

The data comes from the following sources:

* [Multiple-Translation Chinese (MTC) part 1](https://catalog.ldc.upenn.edu/LDC2002T01)
* [Multiple-Translation Chinese (MTC) part 3](https://catalog.ldc.upenn.edu/LDC2004T07)

### Test set

The files that are used as test data to report your performance are in
the `test` directory. There are four references for each source sentence.

The data comes from the following source:

* [Multiple-Translation Chinese (MTC) part 4](https://catalog.ldc.upenn.edu/LDC2006T04)

### Language Model

The language model files are in the `lm` directory.

* `en.gigaword.3g.arpa.gz`: large LM estimated using
  Kneser-Ney smoothing from the [English Gigaword corpus](https://catalog.ldc.upenn.edu/LDC2011T07)
* `en.gigaword.3g.filtered.arpa.gz`: medium size LM filtered from the
  large LM for the dev and test files (52MB compressed)
* `en.tiny.3g.arpa`: tiny LM from the decoding homework

## Scripts

The following are scripts that can be used to create a phrase table with
feature values from source, target and alignment data.

* `pp_xtrct_sc.sh`: shell script to run phrase extractor on the toy
  data set.
* `pp_xtrct.sh`: shell script to run phrase extractor. It splits the
  data into shards of 20K sentences each and then runs phrase extraction
  in parallel. It then filters each phrase file for `dev` or `test`
  and finally merges the phrases for each shard.

The scripts above call the following Python programs in the right
sequence.

* `PPXtractor_ph1.py`: python program for extracting phrase-pairs from
  a source, target, and alignment files.
* `PPXtractor_ph2n3.py`: python program for identifying the source
  phrases in the given dev/test set and filter the phrase file for the
  source phrases.
* `PPXtractor_ph2.py`: python program for computing forward and reverse
  lexical scores.
* `PPXtractor_ph3.py`: python program for estimating the forward
  $$P(s|t)$$ and reverse $$P(t|s)$$ probabilities using relative frequency
  estimation.

