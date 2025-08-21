EPOCHS=500
for DATA in gapfill
do
	for LEVEL in 2
	do  
        	for SEED in 0
        	do  
            		python main_TabTransf_GeoInfo.py $LEVEL $SEED $DATA $EPOCHS > logTabTransf_GeoInfo_${DATA}_Lev${LEVEL}_${EPOCHS}ep_seed${SEED}
        	done
	done
done
