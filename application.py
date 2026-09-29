from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        # Extract and cast inputs safely with correct mapping
        data = CustomData(
            gender=request.form.get('gender', ''),
            race_ethnicity=request.form.get('ethnicity', ''),
            parental_level_of_education=request.form.get('parental_level_of_education', ''),
            lunch=request.form.get('lunch', ''),
            test_preparation_course=request.form.get('test_preparation_course', ''),
            reading_score=int(request.form.get('reading_score', 0)),
            writing_score=int(request.form.get('writing_score', 0))
        )

        pred_df = data.get_data_as_data_frame()
        print("Feature DataFrame for Prediction:\n", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Round the predicted score to 2 decimal places for clean UI output
        prediction_result = round(results[0], 2)

        return render_template('home.html', results=prediction_result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('home.html', error_message="An error occurred while processing your input. Please check your data.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)