# Stone Paper Scissor Game

Stone, Paper, Scissor is a popular hand game usually played by two people, in which both the players form one of the three shapes "Stone" (a closed fist), "Paper" (palm with all fingers outstretched) or "Scissor" (fist with index and middle finger extended to form V shape) simultaneously.
There are two possible outcomes of such a game: tie or a win for one player and lose of another.

In this project, image_data_generate.py is used to generate image data which is stored in image_dataset folder. There are four classes: "Stone", "Paper","Scissor" and "None". A Convolutional Neural Network Model is trained to obtained optimized weights. Later, this trained model is used to create an OpenCV based Game where one player is the user while other player is Computer. 

Model Pruning, a technique for optimizing Neural Network Models, is applied over CNN and its impact on model size, performance and accuracy is also discussed. The SPS_Training_withPruning.ipynb contains the Training and Pruning Algorithms using Tensorflow and also discusses about What pruning is and Why we are using it for this Application.
