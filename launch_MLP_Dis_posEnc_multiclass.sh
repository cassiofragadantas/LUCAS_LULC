EPOCHS=500
for DATA in prime gapfill
do
	for LEVEL in 1 2
	do  
        	for SEED in 0
        	do  
            		python main_MLP_Dis_posEnc_multiclass.py $LEVEL $SEED $DATA $EPOCHS > logOUR_posEncMulti_${DATA}_Lev${LEVEL}_${EPOCHS}ep_seed${SEED}
        	done
	done
done
