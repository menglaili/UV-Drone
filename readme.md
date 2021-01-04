key_code contains codes for controller, data collection, training, evaluation

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

Preparations:

    - install rabbitmq server for connecting webcam and the drone: `sudo apt-get install rabbitmq-server`
    - install required packages:
    ```
    pip install -r requirements.txt
    ```

Starting the simulation:
    ```
    sudo systemctl start firmwared
    sphinx /opt/parrot-sphinx/usr/share/sphinx/drones/anafi4k.drone::stolen_interface=::simple_front_cam=true

    ```
