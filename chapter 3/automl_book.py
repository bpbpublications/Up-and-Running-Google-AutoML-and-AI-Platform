from google.cloud import automl

project_id = 'automlgcp'
model_id = 'TCN14567890876540944'
issue_description = "System Utilization is high on server. Please Suggest"

def classification():

  prediction_obj = automl.PredictionServiceClient()

  # Retrieves path of the model
  model_details = automl.AutoMlClient.model_path(project_id, 'us-central1', model_id)
  issue_snippet = automl.TextSnippet(content=issue_description,mime_type='text/plain')  
  payload_data = automl.ExamplePayload(text_snippet=issue_snippet)
  predicted_response = prediction_obj.predict(name=model_details, payload=payload_data)
  for result_payload in predicted_response.payload:
    print(u'Predicted class name: {}'.format(result_payload.display_name))
    print(u'Predicted class score: {}'.format(result_payload.classification.score))


if __name__=='__main__':
  classification()
