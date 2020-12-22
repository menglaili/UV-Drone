key_code: for reading convenience, I copy important code files into this folder
    - control_collectData
        - Anafi_tracking.py: webcam localization system and localization publisher
        - keyboard_ctrl.py: keyboard controller for Anafi
        - video_control.py: collect reference path, collect control correction, data augmentation using active transfer learning
    - NNtraining
        - data_provider.py: read data from files and feed them to neural network
        - models.py: the neural network architecture
        - NNutils.py: helper functions
        - train.py: main function for training the result
        - evaluation.py: evaluate the network performance and draw plots
    - webcam
        - find_extrinsct.ipynb: find the extrinsct of the webcam for localization

original_files: the original files containing all messy staffs