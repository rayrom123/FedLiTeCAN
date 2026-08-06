# Transformer-based Intrusion Detection System for CAN Network
This is the implementation of an encoder-only Transformer used for intrusion detection in CAN Bus network.
## The datasets used for this implementation are :
1. [Car Hacking](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
2. [Survival Analysis](https://ocslab.hksecurity.net/Datasets/survival-ids)
3. [Car Hacking: Attack & Defense Challenge 2020](https://ocslab.hksecurity.net/Datasets/carchallenge2020)

## Code
1. car_hacking.py: utilizes the transformer model to perform intrusion detection on the car hacking dataset
2. survival_analysis.py: utilizes the transformer model to perform intrusion detection on the survival analysis dataset
3. unseen_attack.py: performs cross-dataset evaluation by training on the survival_analysis dataset and testing on the Car Hacking dataset and Car Hacking: Attack & Defense Challenge 2020 dataset
4. server.py, client.py: utilizes the transformer model to perform intrusion detection using the car hacking and survival analysis dataset in FL settings

## Kaggle quick start for CAN FL-only

This repo has been adapted for the FL-only CAN dataset:

`/kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt/`

Train data:

`federated_data/client_0.pt` ... `client_9.pt`

Global test data:

`global_test_data.pt`

### Train

```bash
%%bash
set -e

rm -rf /kaggle/working/FedLiTeCAN
git clone https://github.com/rayrom123/FedLiTeCAN.git

cd /kaggle/working/FedLiTeCAN

pip install -q flwr scikit-learn pandas

DATA_ROOT="/kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt"

python server_iov.py \
    --mode train \
    --num-clients 10 \
    --rounds 30 \
    --local-epochs 1 \
    --test-file "$DATA_ROOT/global_test_data.pt" \
    --test-max-samples 0 &

sleep 30

for cid in 0 1 2 3 4 5 6 7 8 9; do
    python client_iov.py \
        --client-id $cid \
        --data-root "$DATA_ROOT" \
        --max-samples 0 \
        --connect-retries 120 \
        --retry-wait 5 &
done

wait
```

### Resume

```bash
%%bash
set -e

cd /kaggle/working/FedLiTeCAN

DATA_ROOT="/kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt"

python server_iov.py \
    --mode resume \
    --checkpoint checkpoints_can_fl/round_013.pth \
    --num-clients 10 \
    --rounds 30 \
    --local-epochs 1 \
    --test-file "$DATA_ROOT/global_test_data.pt" \
    --test-max-samples 0 &

sleep 30

for cid in 0 1 2 3 4 5 6 7 8 9; do
    python client_iov.py \
        --client-id $cid \
        --data-root "$DATA_ROOT" \
        --max-samples 0 \
        --connect-retries 120 \
        --retry-wait 5 &
done

wait
```

### Test checkpoint

```bash
%%bash
set -e

cd /kaggle/working/FedLiTeCAN

DATA_ROOT="/kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt"

python server_iov.py \
    --mode test \
    --checkpoint checkpoints_can_fl/round_030.pth \
    --test-file "$DATA_ROOT/global_test_data.pt" \
    --test-max-samples 0
```

Outputs:

- `metrics_can_fl.csv`
- `metrics_can_fl_round.log`
- `checkpoints_can_fl/round_030.pth`
