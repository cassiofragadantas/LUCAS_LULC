for DATA in prime gapfill
do
	for LEVEL in 1 2
	do  
        	for SEED in 0
        	do  
            		python main_SVM.py $LEVEL $SEED $DATA > logSVM_${DATA}_Lev${LEVEL}_seed${SEED}
        	done
	done
done
