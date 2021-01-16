# Disaster Response Pipeline Project

This is project from the Udacity Data Scientist course which to improve my data science/engineer skills.

## Getting started

### Requirements:
- Python (64bit)

### Setup:
1. Clone project `git clone https://github.com/Schayik/u-disaster-response-pipeline.git`
2. Create virtual environment: `python -m venv .venv`
3. Activate virtual environment: `.venv/Scripts/activate`
4. Install dependencies: `pip install -r requirements.txt`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

**Note**: I had memory issues which were resolved by using Python 3.9 64bit instead of Python 3.8 32bit. This article was very helpful: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
