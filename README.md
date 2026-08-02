# CIVFinalProject

**Team Members**
- Ron Agady
- Nada Khan
- Hyun Joon Kim

**Overview**

Our final CVI project implements a self driving car using a Convolutional Neural Network. The model is trained on images collected from the Udacity Self Driving Car Simulator and predicts steering angles in real time for a self driving vehicle to successfully drive on the simulator.

The project includes data augmentation, preprocessing, batch generation, model training, and simulator testing.

**Main technologies**

- Python 3.8
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Udacity Self-Driving Car Simulator

**Approach and Challenges**

&nbsp;&nbsp;&nbsp;&nbsp;**Ron**: There were several challenges encountered during the setup operations and basic model initiation procedure that took a long time to overcome when put all together. One of which was even the basic environment and dependency setup, trying to get a right version of Python 3.8 to link up with the system, gathering the right packages, configuring Registry settings to enable packages. Even then, it was its own headache trying to get the Udacity driving simulator to install due to issues with the Unity engine, and then compile the training data together for my teammates. 

&nbsp;&nbsp;&nbsp;&nbsp;However, the biggest issue I faced so far was trying to create the initial .h5 base model used by my team (not regarding any ML refinement or data augmentation techniques.) This was because I was working now with a regression model that was meant to predict the next move, as opposed to classification which we had learned in class. It took quite a bit of research to figure out how to do so and what were the different variables involved with the operation.

&nbsp;&nbsp;&nbsp;&nbsp;**Hyun Joon**: Hyun Joon: Some of the challenges that I faced during this project was difficult to exactly pinpoint what was causing the failure at the time. The simulator kept crashing for me, which I wasn’t able to resolve for my computer and had to download Unity editor and set it up in a roundabout way by rebuilding it with Unity 2020.3. Once I was able to get the simulator working, it wasn’t communicating with the script, and it turned out that the versions of the python-socketio and python-engineio, they had to be downgraded because of the Socker.IO v1 client used by the Unity Simulator that I had to use. 

&nbsp;&nbsp;&nbsp;&nbsp;Another challenge was that, I wanted to make sure after completing the preprocessing, batching and augmentation scripts that it was running the simulation at least up to this point of the project, but I noticed that, once it connected, it just wasn’t able to drive at all. I had to realize that the model prediction value was actually held constant, and it wasn’t being changed and adjusted at all, at 0.00155544, which indicated that the model collapsed during the training. Therefore, going back to the modelCreator.py, by adjusting from relu to elu and changing the learning rate and the number of epochs, and retraining the model, it was possible to fix the issue and see the automatic car drive successfully and see the values change and adjust.

&nbsp;&nbsp;&nbsp;&nbsp;Next challenge was that, once there was a good portion of track being self driven by the car, there was a point in the map, where it looked like given the height of the bump that continued off the outside barrier, where the grey-concreteness of the barrier disappears, but the stretch of dirt land branches off the track, given model was having a difficult time detecting that it’s not part of the road, thus unable to turn left instead and stay on the track. At that point, I raised the number of epochs to 60, noticing yet a small decrement of losses still near the end of 30 epochs, which was successful in overcoming that specific part of the map it had difficulty detecting and make the right decision.

&nbsp;&nbsp;&nbsp;&nbsp;**Nada**:



**Project Structure**

```text
CVIFinalProject/
│── baseSelfDrivingCarModel.h5    # Pre-trained CNN model
│── modelCreator.py               # Trains the steering model
│── preprocessing.py              # Image loading and preprocessing
│── data_augmentation.py          # Data augmentation functions
│── batch_generator.py            # Batch generation for training
│── TestSimulation.py             # Runs the model in the simulator
│── package_list.txt              # Required Python packages
└── README.md
```

**Dataset**

```text
CVI620NSATestingData/
│── driving_log.csv
└── IMG/
    │── image files
```

The training dataset was too large for GitHub and is hosted separately on Google Drive.

https://docs.google.com/document/d/1VsgC1u_xgc_3gCdXvwtqa4wVcSDBhRy75c2oz4tgfAM/edit?usp=sharing

After downloading, extract the contents into the project directory while preserving the original folder structure.

**Environment & Dependency Setup:**

1.  Clone and pull the **GitHub repo** associated with the final project using the git clone terminal command in any **directory** of your choosing.
    
2.  All training data is located inside a **ZIP folder** on **Google Drive** since **GitHub** could not store it, so download that: [https://drive.google.com/file/d/1fh3LzC87amvD5ca311dL7KVTK5jO72bX/view?usp=drive\_link](https://drive.google.com/file/d/1fh3LzC87amvD5ca311dL7KVTK5jO72bX/view?usp=drive_link)
    
3.  **Unzip** the contents _(Expected 35 to 45 minute wait time, so perform other steps concurrently in the meantime.)_
    
4.  Enable **LongPathEnabled** registry on your **Windows** device. This can be done as either a **Powershell** command or a **RegEdit** change
    
    1.  If done via **RegEdit**, go to Computer\\HKEY\_LOCAL\_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem, and the **LongPathEnabled** DWORD value to 1, which signals to it that it is enabled.
        
    2.  If done as a **Powershell** command, instead open up **Powershell** as an administrator, then copy & paste this command, then execute: New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force 
        
5.  Create a **Python 3.8 virtual environment** using the following command: py -3.8 -m venv "”
    
    1.  If on your better judgement, you can do this on any Python version you choose if running into issues with 3.8. Preferably, **Python 3.8** should be installed before starting this step: [https://www.python.org/downloads/release/python-3810/](https://www.python.org/downloads/release/python-3810/)
        
6.  Install the **Udacity Car Driving simulator Term 1 Beta Simulator** version, located on this link: [s3-us-west-1.amazonaws.com](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip) (**Windows** version)
    
7.  **Unzip** those contents that you installed, and perform a playtest to confirm it is working via launching the app, configuring launch options, then selecting ‘**Manual driving**’ and doing some light amount of driving.
    
8.  **Restart your computer** to have the **LongPathEnabled** changes come into effect. **Do not do this until the Training Data ZIP Folder has finished extracting however - wait if need be.** Having run this command allows us to properly install all packages in the later steps.
    
9.  Return to the **virtual environment** and activate it by running the activate script via a terminal command in the scripts directory of the **virtual environment** you created.
    
10.  Go to the **cloned repository** _(or where_ _**package\_list.txt**_ _should be located,)_ and run the **Powershell** command to install all the **packages** inside the repo’s **package\_list** **file:** Get-Content package\_list.txt | ForEach-Object { pip install $\_ 2>$null } 
    
    1.  Alternatively, you can instead run the terminal command: pip install package\_list.txt, but this does not allow **packages** to be installed concurrently, and any error that occurs from this command will stop the whole process, forcing a redo of reinstallations if running the command again regularly.
        
- Note: If the Udacity simulator connects but the vehicle does not respond, or if no telemetry is received, install these compatible Socket.IO versions inside your virtual environment:

```bash
pip install "python-socketio==4.6.1" "python-engineio==3.13.2"
```
- You may see a warning that flask-socketio requires a newer version of python-socketio. This warning can be safely ignored because this project does not use flask_socketio, using python-socketio 

11.  Once done, everything is now set-up for proper use: You can now run the modelCreator.py file under this **virtual environment** and create a new model.
    
    1.  You can use this terminal command to do this: & "/Scripts/python.exe" "/modelCreator.py"

**Training the Model**

After completing the environment setup and ensuring the dataset is extracted correctly:

1. Activate the Python virtual environment.
2. Ensure the training data directory contains:
   - `driving_log.csv`
   - `IMG/` folder containing the recorded images
3. Run the training script:

```bash
python modelCreator.py
```

**Generated Output**

Running modelCreator.py generates:

- model.h5
- baseSelfDrivingCarModel.h5
- steering_histogram.png
- training_plots.png

**Image Preprocessing**

Each image is cropped, converted to YUV colour space, blurred, resized to 200×66, and normalized before being passed to the CNN.

**Data Augmentation**

Training images are randomly augmented using:

- Horizontal flipping
- Brightness adjustment
- Zoom
- Translation
- Rotation

**Dataset Batching**

The dataset is grouped into smaller batches so the model can train efficiently without loading all images into memory at once. Each batch is processed consistently before training, helping the CNN learn from the prepared data in an organized and memory-efficient way.

**Testing**

After training:

1. Launch the Udacity simulator.
2. Select Autonomous Mode.
3. Run TestSimulation.py.
4. The simulator will connect automatically and use the trained model.
