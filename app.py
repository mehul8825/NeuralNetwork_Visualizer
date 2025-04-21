from flask import Flask, render_template, request, jsonify
import NeuralNetworksModule as Nnm
import numpy as np
import torch

app = Flask(__name__)

dataset = {'X': [], 'y': []}
model = None
mpp = {'A': 0, 'B': 1, 'C': 2}
mp = {0: 'A', 1: 'B', 2: 'C'}
shape = (500, 500, 3)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_point', methods=['POST'])
def add_point():
    data = request.json
    x = data['x']
    y = data['y']
    label = data['label']
    dataset['X'].append([x, y])
    dataset['y'].append(mpp[label])
    return jsonify(success=True)


@app.route('/train', methods=['POST'])
def train_model():
    global model
    model = Nnm.MyNN(dataset)
    model.Training()
    return jsonify(success=True)


@app.route('/get_loss', methods=['GET'])
def get_loss():
    if model:
        print("LOSS VALUES:", model.loss_values)  # <-- add this
        return jsonify(loss=model.loss_values)
    return jsonify(loss=[])


@app.route('/predict_canvas', methods=['GET'])
def predict_canvas():
    if not model:
        return jsonify(image=[])

    x_values = np.arange(shape[1])
    y_values = np.arange(shape[0])
    xx, yy = np.meshgrid(x_values, y_values)
    xy_pairs = np.column_stack([xx.ravel(), yy.ravel()])
    xy_tensor = torch.tensor(xy_pairs, dtype=torch.float32) / shape[0]

    with torch.no_grad():
        predictions = model.model(xy_tensor)
        predicted_classes = predictions.argmax(dim=1).numpy()
        predicted_classes = predicted_classes.reshape(shape[:2])
    return jsonify(predicted=predicted_classes.tolist())


if __name__ == '__main__':
    app.run(debug=True)
