from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import multiprocessing as mp
import time
import os

app = Flask(__name__)
socketio = SocketIO(app)

processes = {}
process_status = {}


def train_process(config, process_id):
    # 模拟训练过程
    for epoch in range(config['num_epochs']):
        if process_status[process_id] == "stopped":
            break
        time.sleep(1)  # 模拟训练
        process_status[process_id] = f"running: epoch {epoch+1}/{config['num_epochs']}"
        socketio.emit('training_status', {
            'process_id': process_id,
            'status': process_status[process_id]
        })
    process_status[process_id] = "finished"
    socketio.emit('training_status', {
        'process_id': process_id,
        'status': process_status[process_id]
    })


@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json
    process_id = str(len(processes) + 1)
    process_status[process_id] = "starting"
    config = data['config']
    p = mp.Process(target=train_process, args=(config, process_id))
    p.start()
    processes[process_id] = p
    process_status[process_id] = "running"
    return jsonify({"process_id": process_id})


@app.route('/stop_training/<process_id>', methods=['POST'])
def stop_training(process_id):
    if process_id in processes:
        process_status[process_id] = "stopped"
        processes[process_id].terminate()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "process not found"}), 404


@app.route('/status/<process_id>', methods=['GET'])
def get_status(process_id):
    if process_id in process_status:
        return jsonify({"status": process_status[process_id]})
    return jsonify({"error": "process not found"}), 404


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
