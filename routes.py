from flask import Flask, request, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

# Dummy data and status for the sake of the example
simulation_data = {"value": 0}
simulation_status = {"status": "stopped"}

@app.route('/upload-data', methods=['POST'])
def upload_data():
    # Specify the directory where you want to save the data
    save_directory = '/home/vtiyyal1/projectml/'
    os.makedirs(save_directory, exist_ok=True)

    # Extract data from the request
    data = request.json
    filename = data.get('filename', 'default.json')  # You can pass a filename in your request

    # Save data to a file in the specified directory
    file_path = os.path.join(save_directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file)

    return jsonify({"message": "Data uploaded successfully"}), 200

@app.route('/simulation-start', methods=['POST'])
def start_simulation():
    # Extract data from the request
    data = request.json
    user_demographics = data.get('user_demographics')
    tweet_data = data.get('tweet_data')
    news_data = data.get('news_data')
    policies = data.get('policies')

    # Add your simulation logic here
    # You now have user_demographics, tweet_data, news_data, and policies to use in your simulation

    return jsonify({"message": "Simulation started with provided data"}), 200


@app.route('/get-json-file', methods=['GET'])
def get_json_file():
    file_directory = '/mnt/hwfile/chenzhi/ABE/SAMOYEDS/'#'/home/vtiyyal1/projectml'
    filename = 'vj_json.json'

    file_path = os.path.join(file_directory, filename)
    if os.path.exists(file_path):
        return send_from_directory(file_directory, filename)
    else:
        return jsonify({"error": "File not found"}), 404
    
"""
@app.route('/get-pkl-file', methods=['GET'])
def get_pkl_file():
    file_directory = '/home/vtiyyal1/projectml'
    filename = 'combined_posts_texts_covid.pkl'

    file_path = os.path.join(file_directory, filename)
    if os.path.exists(file_path):
        return send_from_directory(file_directory, filename)
    else:
        return jsonify({"error": "File not found"}), 404
"""

@app.route('/simulation-status', methods=['GET'])
def get_simulation_status():
    return jsonify(simulation_status)



@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(simulation_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
