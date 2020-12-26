**to have the drone do one piloted path and then have the drone follow it 5 times. Now I don’t even need for the webcam to save and allow me to compute the error). Can you help me by writing a simple summary for the following steps:**

**c. Can you tell me which .py file should i run to start reference flight?  **

- ```
  cd control_collectData
  ```

- Modify the *video_control.py* file: 

  1. Set the reference path: 

  ```python
  if __name__ == "__main__":
      path = [(0.5, 0), (0.5, 0), (0.5, 0)] 
      # (distance, turning angle) between each checkpoints
  ```

  2.  Choose proper IP address: if physical drone, it is writen on Anafi's body

  ```python
  if __name__ == "__main__":
      path = [(0.5, 0), (0.5, 0), (0.5, 0)] 
  		with olympe.Drone("192.168.42.1") as drone: # if simulation, use "10.202.0.1"
  ```

  3. Choose the proper function:  `streaming_example.get_dis_ref()`

  ```python
  if __name__ == "__main__":
      path = [(0.5, 0), (0.5, 0), (0.5, 0)] 
  		with olympe.Drone("192.168.42.1", loglevel=0) as drone: # "192.168.42.1" "10.202.0.1"
          streaming_example = StreamingExample(drone, True, path)
          # streaming_example.dis2pair()
          print(streaming_example.path)
          streaming_example.start()
          # streaming_example.fly_dis_traj()   should comment out others
          streaming_example.get_dis_ref()  # <------- modify this line
          streaming_example.stop()
          streaming_example.postprocessing()
  ```

  4. choose which folder to store the data: modify the input of `get_dis_ref()`

  ```python
  streaming_example.get_dis_ref(reference_file_dir) # if not enter anything, the reference data is store in './reference_data'
  ```

  5. In the Olympe virtual environment, run: `python3 video_control.py`

  6. Take Off: press `t`

  7. If WebCam is running, the terminal will print the distance needed to move; If no Webcam, the terminal will print nothing

  8. Control the drone to move: press `w` forward, `a` leftward, `d` rightward, `s` backward

  9. Either you press `c` , or the Webcam has detected that the drone comes to the nominal checkpoint, the drone will automatically store the current FPV camera image

  10. The terminal will give a hint for the next planned action

  11. After all checkpoint is finished, the drone will landed automatically. Or if any emergency happens, press `l` to land immediately

  12. Program exit

  

- **side question:**

  In the `reference_file_dir` folder or by default `'./reference_data'`, image data will store in `/image`

  

**d. Can you tell me after the reference flight is over, which file do i need to run (assuming I have returned the drone to the start position) so it will take over and fly the drone according to what it sees (and comparisons w reference)? **

​	This function is included in *video_control.py*, *StreamingExample* class, *test_NN_perform()* function; For convenience, the code now for data augmentation and testing are blended, so this function needs the Webcam to run although the Webcam here does not help testing.

- Modify the *video_control.py* file: 

  same with previous step 1. & step 2.

  3. Choose the proper function:  `streaming_example.test_NN_perform()`

  ```python
  if __name__ == "__main__":
      path = [(0.5, 0), (0.5, 0), (0.5, 0)] 
  		with olympe.Drone("192.168.42.1", loglevel=0) as drone: # "192.168.42.1" "10.202.0.1"
          streaming_example = StreamingExample(drone, True, path)
          # streaming_example.dis2pair()
          print(streaming_example.path)
          streaming_example.start()
          streaming_example.test_NN_perform()  # <------- modify this line
          streaming_example.stop()
          streaming_example.postprocessing()
  ```

  4. choose which folder to store the data: modify the input of `test_NN_perform()`

  ```python
  streaming_example.test_NN_perform(augmented_data_file) # if not enter anything, the data is store in "./augmented_training_data"
  ```

  5. In the Olympe virtual environment, run: `python3 video_control.py`

  6. Press `t` to take off
  7. If any emergency happens, continuously press `l`  until it begins to land, otherwise the drone will automatically fly and do correction according to the trained model



**When I have a webcam,**

​	**a. I should run the find_extrinsct.ipynb and save trans_mat. To do so, I need to put in some number of objects (In the the jupyternotbook assumes there are 6 objects, is that true?). I understand I get image_points from the image I get; how do I get model_points?**

​	You can use as many points as you want, but 6 points is enough.  

​	The model_points is the physical point's coordinates measured in real world. First choose an origin (0, 0, 0) and X-Y-Z frame's direction in your space. Second spread some markers in different areas with different height but make sure that the Webcam can cover these markers. Third, use ruler to measure the X-Y-Z coordinate of each marker in the frame you choose. Last, put these coordinates (x, y, z) in to `model_points`



​	**b. Now I am ready to run the webcam w Anafi_tracking.py, right?  Do I run this in parallel with the steps in 1 above?  **

​	Yes, run this file in a different terminal.



​	**b. i. where do the results get saved? **

​	The trajectory of the drone is not saved by `Anafi_tracking.py`. There is function in `video_control.py` to store the pose information. By running this file, a GUI showing the Webcam will automatically bump out. 



​	**b. ii. do you have a post flight analytics that can visualize the path and compute MSE like you did on slide 8 of your presentation? **

​	I didn't store this part in my code. In `evaluation.py` file, it store the function to draw a cumulative histogram plot. After the pose data is stored by `video_control.py`, we can read from the pose data text file to draw the plot and calculate the MSE.



