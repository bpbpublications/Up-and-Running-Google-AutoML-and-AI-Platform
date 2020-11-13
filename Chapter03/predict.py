from google.cloud import automl
from flask import Flask, request
import json

app=Flask(__name__)

@app.route ("/sentiment", methods=['POST'])
def predictSentiment():
    try:
        json_data = request.get_json(force=True)
        project_id = json_data["project_id"]
        model_id = json_data["model_id"]
        sentence = json_data["content"]
        result = []
        sentiment_prediction_model = automl.PredictionServiceClient()
        # Retrieves path of the model
        model_details = automl.AutoMlClient.model_path(project_id, 'us-central1', model_id)
        sentence_snippet = automl.TextSnippet( content=sentence,mime_type='text/plain')  
        payload_data = automl.ExamplePayload(text_snippet=sentence_snippet)
        predicted_response=sentiment_prediction_model.predict (name=model_details,    payload=payload_data)
        for payload_result in predicted_response.payload:
          result= "Predicted sentiment score: {}"   .format(payload_result.text_sentiment.sentiment)
        return (result)
    except Exception as e:
      return {"Error": str(e)}

if __name__ == "__main__" :
  app.run(port="5000")
