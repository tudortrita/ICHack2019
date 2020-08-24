from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
import base64
from imageio import imread
import io
import scipy
import scipy.misc


import numpy as np

app = FlaskAPI(__name__)


# Note, request should take form similar to:
#curl -X 'POST' -d '{"image_data": "infor"}' -H "Content-Type: application/json" <URL-HERE>:5000


@app.route("/", methods=['POST'])
def image_request():
    """
    Obtain image and process
    """
    if request.method == 'POST':
        note = str(request.data.get('image_data'))
        image_data = base64.b64decode(note)
        im = imread(io.BytesIO((image_data)))
        im = np.asarray(im)
        im_shape = im.shape # (3120, 4160, 3)
        if im_shape[0] > im_shape[1]:
            new_im = im[0:im_shape[1], 0:im_shape[1], :]
        elif im_shape[0] < im_shape[1]:
            new_im = im[0:im_shape[0], 0:im_shape[0], :]
        else:
            new_im = im
        new_im = scipy.misc.imresize(new_im, [100,100])
        with open('/homes/kgs13/biomedic/projects/ichack2019/ichack2019/picture.jpeg', 'wb') as f:
            f.write(image_data)
        with open('/homes/kgs13/biomedic/projects/ichack2019/ichack2019/shape.txt', 'w') as f:
            f.write(str(im_shape))

        scipy.misc.imsave('/homes/kgs13/biomedic/projects/ichack2019/ichack2019/newpicture.jpeg', new_im)
        with open('/homes/kgs13/biomedic/projects/ichack2019/ichack2019/new_shape.txt', 'w') as f:
            f.write(str(new_im.shape))

        return 'success', 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
