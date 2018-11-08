#!/bin/bash

rm build.err build.logs model-file out-file eval.logs
source venv/bin/activate
pip install -r requirement.txt
python build_tagger.py sents.train model-file 2> build.err > build.logs
python run_tagger.py sents.test model-file out-file > run.logs
python eval.py out-file sents.answer > eval.logs