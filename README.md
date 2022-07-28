# Disaster Response Pipeline Project

## Project Description
	This project aims to create a machine learning pipeline to classifies disaster messages. The datasets contain real messages that were sent disater events. The model that has been built in this project categorizes these events so that the messages can be sent to an appropriate disaster relief agency. The data was obtained from Appen (formally Figure 8) to build an ETL and Machine Learning pipeline.
	The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
    
## File Description
- app
| - template
| |- master.html  
| |- go.html  
|- run.py  

- data
|- disaster_categories.csv  
|- disaster_messages.csv  
|- process_data.py
|- DisasterResponse.db   

- models
|- train_classifier.py
|- classifier.pkl 

- README.md


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/

### References
Udacity Data Science Nanodegree-Data Engineering course, Figure 8, GitHub, Stack Overflow. 