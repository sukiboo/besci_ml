
import numpy as np
import pandas as pd
import tensorflow as tf
import shap


def generate_data(num, seed):
    '''simulate synthetic coin flips'''
    x, y = [], []
    np.random.seed(seed)
    for _ in range(num):

        # generate one-hot encoded features
        shape = np.eye(len(shapes))[np.random.randint(len(shapes))]
        color = np.eye(len(colors))[np.random.randint(len(colors))]
        material = np.eye(len(materials))[np.random.randint(len(materials))]
        flip = np.random.randint(2, size=1+np.random.randint(100)).mean()

        # save generated data point
        x.append(np.concatenate([shape, color, material]))
        y.append(flip)

    return np.array(x), np.array(y)


def truth_data():
    '''ground truth coin flips'''
    x, y = [], []
    for s in range(len(shapes)):
        for c in range(len(colors)):
            for m in range(len(materials)):
                shape = np.eye(len(shapes))[s]
                color = np.eye(len(colors))[c]
                material = np.eye(len(materials))[m]
                x.append(np.concatenate([shape, color, material]))
                y.append(0.5)
    return np.array(x), np.array(y)


# simulation parameters
shapes = ['round', 'octagonal', 'hexagonal', 'square', 'triangle']
colors = ['red', 'green', 'blue', 'yellow', 'black']
materials = ['metal', 'wood', 'plastic', 'glass']
seed = 2022


# generate train and test data
x_tr, y_tr = generate_data(10000, seed)
x_ts, y_ts = truth_data()

df_tr = pd.DataFrame(x_tr, columns=(shapes + colors + materials))
df_tr['flip'] = y_tr
df_ts = pd.DataFrame(x_ts, columns=(shapes + colors + materials))
df_ts['flip'] = y_ts


# create model
tf.random.set_seed(seed)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=x_tr[0].size),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
model.fit(x_tr, y_tr, batch_size=32, epochs=100, verbose=0)
model.evaluate(x_ts, y_ts)

# compute and plot shapley values
explainer = shap.Explainer(lambda x: model.predict(x).reshape(-1), x_tr)
shap_values = explainer(x_ts)
shap_values.feature_names = shapes + colors + materials
shap.summary_plot(shap_values)
shap.plots.waterfall(shap_values[0])

