import base64
import json
from flask import Flask, render_template, request
from flask_cors import CORS
import os
from worker import speech_to_text, text_to_speech, watsonx_process_message

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    # Render the index.html template (you should have an index.html file in the templates folder)
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("Processing Speech-to-Text")

    # Get the user's speech from their request
    audio_binary = request.data

    # Call the speech_to_text function to transcribe the speech
    text = speech_to_text(audio_binary)

    # Return the response to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )

    print(response)
    return response


@app.route('/process-message', methods=['POST'])
def process_message_route():
    # Get user's message and preferred voice from their request
    user_message = request.json['userMessage']
    print('user_message:', user_message)
    voice = request.json.get('voice', '')

    # Call watsonx_process_message function to process the user's message and get a response back
    watsonx_response_text = watsonx_process_message(user_message)

    # Clean the response to remove any empty lines
    watsonx_response_text = os.linesep.join([s for s in watsonx_response_text.splitlines() if s])

    # Call the text_to_speech function to convert Watsonx API's response to speech
    watsonx_response_speech = text_to_speech(watsonx_response_text, voice)

    # Convert watsonx_response_speech to base64 string so it can be sent back in the JSON response
    watsonx_response_speech = base64.b64encode(watsonx_response_speech).decode('utf-8')

    # Send a JSON response back to the user containing their message's response in both text and speech formats
    response = app.response_class(
        response=json.dumps({
            "watsonxResponseText": watsonx_response_text,
            "watsonxResponseSpeech": watsonx_response_speech
        }),
        status=200,
        mimetype='application/json'
    )

    print(response)
    return response


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
