
<img width="1275" height="477" alt="530187843-62cbbe76-17dc-42a8-a427-8fb084674029" src="https://github.com/user-attachments/assets/64ed7a6a-df71-4f5a-86cd-a7c826cf0d7d" />



# Iris_app


https://github.com/user-attachments/assets/31bbd884-ba13-464f-adc3-1e8a98f5454c





* Project goal:
The objective of this project was to build an interactive web application that allows users to predict flower species (Setosa, Versicolor or Virginica). I wanted to deploy a machine learning model into a functional web application.

* Methodology:
Pipeline: I used the Pandas library to transform the raw dataset and realised a mapping on the numerical variables to their species names to read my data better.
I also created plotly vizuals to create a dynamic visualization tool, allowing users to select different axes and see how the species cluster based on their characteristics.

* Model: I implemented a random model classifier for this multi-class prblem because it manages feature variance well. I also paid attention to efficiency by using Streamlit's caching system, ensuring that the model and data trainings do not restrict the app's performance during user interaction.
Interface: The UI was built with Streamlit, uncluding a sidebar on the edge for user inputs and a main dashboard for data diplay and results.

* Key features:
-Real-time Predictions: Users can adjust sepal and petal dimensions via sliders to see instant classification results.
-Confidence Metrics: The app displays the probability distribution for each species, showing the model’s level of certainty.
-Interactive EDA: A built-in feature explorer to visualize how different measurements separate the three iris species.


