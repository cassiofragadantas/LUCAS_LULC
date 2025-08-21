for DATA in gapfill
do
	for LEVEL in 1 2
	do  
        	for SEED in 0
        	do  
            		python main_TabPFN.py $LEVEL $SEED $DATA > logTabPFN_${DATA}_Lev${LEVEL}_seed${SEED}
        	done
	done
done
