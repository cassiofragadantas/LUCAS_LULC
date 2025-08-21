EPOCHS=500
for DATA in gapfill
do
	for LEVEL in 1 2
	do  
        	for SEED in 0
        	do  
            		python main_TabTransf.py $LEVEL $SEED $DATA $EPOCHS > logTabTransf_${DATA}_Lev${LEVEL}_${EPOCHS}ep_seed${SEED}
        	done
	done
done
