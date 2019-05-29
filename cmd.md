python3 main.py --num_shift=3 --lr=1e-4

python3 main.py --test --snapshot="./snapshot/shift_temp[sh3lr4]_COLLECTIONS_cross_20190523_101806/model_state_epoch_0.th" --num_shift=3

python3 main.py --test --snapshot="./snapshot/shift_temp[sh5lr4]_COLLECTIONS_cross_20190523_102051/model_state_epoch_0.th" --num_shift=5


# HIVE
python3 main.py --num_shift=3 --lr=1e-4 --model="shift_temp" --dataset="HIVE"

# bt test
python3 main.py --bt --bt_path="shift_temp_HIVE_cross_20190523_134454" --model="shift_temp" --dataset="HIVE" --num_shift=3

python3 main.py --bt --bt_path="shift_temp[sh5]_HIVE_cross_20190523_141159" --model="shift_temp" --dataset="HIVE" --num_shift=5


# COLLECTIONS
python3 main.py --num_shift=3 --spans="1,2,3" --lr=1e-4 --model="mspan_temp" --dataset="COLLECTIONS"
python3 main.py --bt --bt_path="mspan_temp_COLLECTIONS_cross_20190524_165650" --model="mspan_temp" --dataset="COLLECTIONS" --num_shift=3

python3 main.py --num_shift=3 --spans="1,2,3" --lr=1e-4 --model="mspan_temp" --dataset="HIVE"
python3 main.py --bt --bt_path="mspan_temp_HIVE_cross_20190524_183227" --model="mspan_temp" --dataset="HIVE" --num_shift=3


COLLECTION(triplet): mspan_temp_COLLECTIONS_cross_20190524_165650
HIVE(triplet): mspan_temp_HIVE_cross_20190524_183227