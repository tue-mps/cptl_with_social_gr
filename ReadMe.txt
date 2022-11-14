Continual Pedestrian Trajectory Learning with Social Generative Replay
    This is a repository about the continual learning for pedestrian trajectory prediction

Requirements
    The current version of the code has been tested with
        - python 3.6.13
        - torch 1.7.1
        - torchvision 0.8.2
    You can run 'conda install --yes --file requirements.txt' to install all dependencies.

Run Individual Learning (IL)
    - You can run 'python main.py --method=batch_learning --log_dir=ETH --dataset_name=ETH', the best model will be saved after the training is completed,
    - Then you can run 'python evaluate_batch_learning.py --log_dir=ETH --dataset_name_train=ETH --dataset_name_test=ETH' to get the results.
    - If you want to change dataset, you can change the 'dataset_name'.

Run CL-NR, CL-ER, CL-CGR and CL-SGR
    Individual experiments can be run with main.py and use the default in code, and all results will be saved to ''./results'.
        - You can 'run python main.py --replay=none' for CL-NR.
        - You can run 'python main.py --replay=exemplars' for CL-ER, in addition, you can change the percentage in './helper/memory_eth.py, ./helper/memory_eth_ucy.py, ./helper/memory_eth_ucy_ind.py'
        - You can run 'python main.py --replay=generative --replay_model=condition' for CL-CGR
        - You can run 'python main.py --replay=generative' for CL-SGR

Run ablation experiments
    - You can run 'python main.py --replay=generative' for CL_SGR_STR, you need to go to the './ablation/CL-SGR-STR' folder.
    - You can run 'python main.py --replay=generative' for CL_SGR_CPD, you need to go to the './ablation/CL-SGR-CPD' folder.
    - You can run 'python main.py --replay=generative' for CL_SGR_RIN, you need to go to the './ablation/CL-SGR-RIN' folder.

Run different task orders
    - First, you need to change the task order in 'main.py' line 177, 178, 179, then you can run 'python main.py --replay=none' and 'python main.py --replay=generative' to get the results of Table 3.
    - Our default task order is 'ETH -> UCY -> inD -> INTERACTION'

Details
    If you want to try another parameter. Main options are:
        --replay: whether use generative replay model? (none | generative)
        --replay_model: the generative replay model (lstm | condition)
        --main_model: determine which main model to choose. (lstm | gat, default is lstm)
        --iters: the number of epochs. (default is 400)
        --z_dim: the dimension of hidden vary z in VAE. (default is 200)
        --batch_size: the mini batch size of current dataset. (default is 64)
        --replay_batch_size: the mini batch size of previous dataset. (default is 64)
        --lr: the learning rate. (default is 0.001)
        --pdf: whether save the results to a pdf.
        --visdom: on-the-fly plots during training.
        --val: whether use the val dataset to select model, default=True.
        --val_class: use the val dataset of current task to validation model.
    The above parameters are applicable to both GR and SGR. For example, you can run 'python main.py --replay=none --main_model=lstm --iters=400 --z_dim=200 --batch_size=64 --replay_batch_size=64 --lr=0.001 --lr_gen=0.001 --val --val_class=current --pdf --visdom', this code will work.

Contact
    If you have any problems, you can contact wuya@gmail.com or a.bighashdel@tue.nl.

Reference
    Our codes borrow some ideas from continual-learning, thanks for their work.
    - van de Ven, Gido M., Hava T. Siegelmann, and Andreas S. Tolias. "Brain-inspired replay for continual learning with artificial neural networks." Nature communications 11.1 (2020): 1-14.