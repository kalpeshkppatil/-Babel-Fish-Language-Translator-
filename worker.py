# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods


# placeholder for Watsonx_API and Project_id incase you need to use the code outside this environment
PROJECT_ID = "skills-network"

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

model_id = ModelTypes.FLAN_UL2

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)


def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP API URL
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = base_url + '/speech-to-text/api/v1/recognize'

    # Set up parameters for our HTTP request
    params = {
        'model': 'en-US_Multimedia',  # Adjust the model parameter if needed
    }

    # Send a POST request with the audio binary data
    response = requests.post(api_url, params=params, data=audio_binary, headers={"Content-Type": "audio/flac"}).json()

    # Initialize text as 'null'
    text = 'null'

    # Check and parse the response to get transcribed text
    if response.get('results'):
        print('Speech-to-Text response:', response)
        text = response['results'][0]['alternatives'][0]['transcript']
        print('Recognized text:', text)

    return text


import requests

def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP API URL
    base_url = 'https://sn-watson-tts.labs.skills.network'
    api_url = base_url + '/text-to-speech/api/v1/synthesize'

    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "?voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }

    # Send a HTTP POST request to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)

    # Log the response status for debugging purposes
    print('Text-to-Speech response status:', response.status_code)
    print('Text-to-Speech response headers:', response.headers)

    # Return the audio content from the response
    return response.content


def watsonx_process_message(user_message):
    prompt = f"""You are an assistant helping translate sentences from English into Spanish.
    Translate the query to Spanish: ```{user_message}```."""
    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)
    return response_text
