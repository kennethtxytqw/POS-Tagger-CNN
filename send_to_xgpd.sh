#!/bin/bash
echo "Please indicate a taskname:"
read taskname
xgpd=xgpd8

ssh sunfire /bin/rm -r $taskname
ssh sunfire ssh $xgpd rm -r $taskname

ssh sunfire mkdir $taskname
echo "Created directory: $taskname"
scp build_tagger.py eval.py Pos_Tagger.py run_tagger.py sents.answer sents.test sents.train runxgpd.sh requirement.txt get_stats.sh sunfire:$taskname/
ssh sunfire scp -r $taskname $xgpd:./

echo "Transferred to $xgpd." 
ssh -t sunfire "ssh $xgpd 'cd $taskname && screen -L -m -d -S $taskname ./runxgpd.sh'"
echo "Started $taskname in $xgpd"