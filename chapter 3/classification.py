from google.cloud import automl
from flask import Flask, request
import json

app=Flask(__name__)

@app.route ("/classification", methods=['POST'])
def Classification():
    try:
        json_data = request.get_json(force=True)
        project_id = json_data["project_id"]
        location = json_data["location"]
        model_id = json_data["model_id"]
        issue_description = json_data["content"]
        result = []
        prediction_obj = automl.PredictionServiceClient()
        model_details = automl.AutoMlClient.model_path(project_id, location, model_id)
        issue_snippet = automl.TextSnippet(content=issue_description,mime_type='text/plain')
        payload_data = automl.ExamplePayload(text_snippet=issue_snippet)
        predicted_response = prediction_obj.predict(name=model_details, payload=payload_data)
        classification = {}
        for result_payload in predicted_response.payload:
            classification["Class_Name"] = result_payload.display_name
            classification["Class_Score"] = result_payload.classification.score
            result.append(classification)
        result = {"results" : result}
        result = json.dumps(result)
        return result
    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__" :
    app.run(port="5000")
