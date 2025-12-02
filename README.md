# CMPT310_Group_2
Final project of Group 2 for the CMPT 310 D100 Fall 2025 course.

This is an implementation of a program that works as a playlist recommendation system. We take an input from the user in the form of a song or playlist and output a playlist of songs that align with the user’s music taste. We use a KNN algorithm and compare the user’s listening preferences with a dataset consisting of a variety of songs and their audio features, then generate recommended songs by finding the best-matched neighbours.

If you just want to run the application you can run the GUI.py by command: "python GUI.py"

## Source Files

### KNNModel.py
This is the KNN model that our program uses to find similar songs.

### Evaluation.py
This program performs 5-fold cross-validation and gives the best hyperparameters, then tests on the final 10%.
You can run it with:
"python Evaluation.py"

### testApp
This is the first tester application we made before upgrading it to GUI.py.
We do not recommend running this application because it is outdated and contains some bugs.

### GUI.py
This is our finalized application that uses the model and displays songs similar to the ones you provide.
To run it:
"python GUI.py"
Sometimes you may need to run it twice for it to work.

## Data Files

### Data.csv
Our main dataset used to train the model. It contains over 170,000 songs with various information about each song.

### Final_Test_Data.csv
The test data collected from real users to evaluate whether the model works in real-world scenarios.

### optimization_result.csv
The data we obtained from running Evaluation.py.

