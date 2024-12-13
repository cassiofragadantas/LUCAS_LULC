EPOCHS=300
for DATA in prime gapfill
do
	for LEVEL in 1 2
	do  
        	for SEED in 0
        	do  
            		python main_XGBoost.py $LEVEL $SEED $DATA $EPOCHS > logXGBoost_${DATA}_Lev${LEVEL}_${EPOCHS}ep_seed${SEED}
        	done
	done
done
