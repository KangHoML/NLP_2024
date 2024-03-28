python3 train.py --data_path "../../datasets/fra_eng.txt" \
                 --model "Seq2Seq" \
                 --hidden_size 256 \
                 > ./result/Seq2Seq.txt

python3 translate.py > ./result/Translation.txt