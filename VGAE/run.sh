for (( k = 10; k <= 20; k++));
do
    echo "K = $k ================================================================="
    python run_SEDR_10x_Genomics_Visium.py --k $k --epochs 1
done
