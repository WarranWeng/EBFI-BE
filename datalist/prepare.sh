## example use for generating datalist

python datalist/generate_datalist.py --mode 2 --data_path /path/to/h5 --portion 0.9 \
                                        --train_txt_name train.txt  \
                                        --valid_txt_name valid.txt

python datalist/generate_datalist.py --mode 1 --data_path /path/to/h5 \
                                        --num 4 --valid_num 2 \
                                        --train_txt_name train.txt  \
                                        --valid_txt_name valid.txt

python datalist/generate_datalist.py --mode 0 --data_path /path/to/h5 \
                                        --num 2  \
                                        --train_txt_name train.txt 
