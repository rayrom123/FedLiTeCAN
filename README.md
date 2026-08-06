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
