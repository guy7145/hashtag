import os
import random
import string

from PIL import Image
from flask import Flask, request

from src.backend.session import Session

NO_LABEL = 'no_label'
NO_ID = -1
MNIST = 'mnist'
SIGN = 'sign-mnist'
dataset = MNIST

app = Flask(__name__, static_url_path='/static')
os.makedirs(app.static_folder, exist_ok=True)
session = Session(dataset)


@app.route('/')
def index():
    return 'pasten'


@app.route('/image/<fid>')
def image(fid):
    fid = int(fid)

    sample = session.oracle.X[fid]

    if dataset == MNIST:
        sample = sample.reshape(8, 8) * 256

    elif dataset == SIGN:
        sample = sample.reshape(28, 28) * 256

    im = Image.fromarray(sample).resize((256, 256)).convert("L")
    name = 'tmp' + ''.join((random.choice(string.ascii_lowercase) for _ in range(10))) + '.jpg'
    im.save(os.path.join(app.static_folder, name))
    return app.send_static_file(name)


@app.route('/next')
def next_image():
    next_id = str(session.next_id())
    print(f'next chosen is {next_id}')
    return next_id


@app.route('/accuracy')
def accuracy():
    return str(session.estimate_accuracy())


@app.route('/oracle')
def oracle():
    sample_id = request.args.get('id', default=NO_ID, type=int)
    label = request.args.get('tag', default=NO_LABEL, type=str)
    print(f'{sample_id} tagged as "{label}"')

    if dataset == SIGN:
        label = ord(label.lower()) - ord('a')

    elif dataset == MNIST:
        label = int(label)

    session.take_label(sample_id, label)
    return 'got it'


@app.route('/hint/<fid>')
def hint(fid):
    fid = int(fid)
    prediction = int(session.predict(sample_id=fid))
    if dataset == SIGN:
        prediction = chr(prediction + ord('A'))

    prediction = str(prediction)

    print(f'hint for {fid} is {prediction}')
    return prediction


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
