SMETimes code is being sorted out.


#preprocess timestamps

python ./preprocess.py --gpu 0 --dataset ETTh1

python ./preprocess_opt.py --gpu 0 --dataset ETTh1

python ./preprocess_gpt.py --gpu 0 --dataset ETTh1

#long-term forecasting

bash ./scripts/long_time_series_forecasting/SMETimes_ETTh1.sh
