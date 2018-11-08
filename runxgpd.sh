#!/bin/bash

rm build.err build.logs model-file out-file eval.logs
python3.5 build_tagger.py sents.train model-file > build.logs
python3.5 run_tagger.py sents.test model-file out-file > run.logs
python3.5 eval.py out-file sents.answer > eval.logs