from flask import Flask, jsonify

app = Flask(__name__)

# Dummy data and status for the sake of the example
simulation_data = {"value": 0}
simulation_status = {"status": "stopped"}

@app.route('/simulation-status', methods=['GET'])
def get_simulation_status():
    return jsonify(simulation_status)

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

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(simulation_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)