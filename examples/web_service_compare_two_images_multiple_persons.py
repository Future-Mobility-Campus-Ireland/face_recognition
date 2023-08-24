import os
import face_recognition
from flask import Flask, request, redirect, render_template
from flask_bootstrap import Bootstrap
from PIL import Image, ImageDraw

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
bootstrap = Bootstrap(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compare_faces(face_encodings1, face_locations1, face_encodings2, face_locations2):
    results = []

    for encoding1, location1 in zip(face_encodings1, face_locations1):
        face1_result = {
            "location": location1,
            "is_match": False
        }

        for encoding2, location2 in zip(face_encodings2, face_locations2):
            is_match = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.45)[0]

            if is_match:
                face1_result["is_match"] = True
                face1_result["matching_location"] = location2

        results.append(face1_result)

    return results


def draw_boxes_on_faces_and_save_image(image, face_locations, path):
    # Convert image to PIL Image object
    pil_image = Image.fromarray(image)

    # Create Draw object for drawing bounding boxes
    draw = ImageDraw.Draw(pil_image)

    # draw box on image
    # Draw bounding box on the matched faces
    for location in face_locations:
        top, right, bottom, left = location
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)

    # Save the modified images with bounding boxes
    output_image_path = os.path.join("static", path)

    pil_image.save(output_image_path)


def compare_faces_in_images(file1, file2, method):
    image1 = face_recognition.load_image_file(file1)
    image2 = face_recognition.load_image_file(file2)

    face_locations1 = face_recognition.face_locations(image1, model=method)
    face_encodings1 = face_recognition.face_encodings(image1, face_locations1)

    face_locations2 = face_recognition.face_locations(image2, model=method)
    face_encodings2 = face_recognition.face_encodings(image2, face_locations2)

    faces1 = compare_faces(face_encodings1, face_locations1, face_encodings2, face_locations2)
    faces2 = compare_faces(face_encodings2, face_locations2, face_encodings1, face_locations1)

    match_faces1_locations = [face["location"] for face in faces1 if face["is_match"]]
    match_faces2_locations = [face["matching_location"] for face in faces1 if face["is_match"]]

    draw_boxes_on_faces_and_save_image(image1, match_faces1_locations, file1.filename)
    draw_boxes_on_faces_and_save_image(image2, match_faces2_locations, file2.filename)

    image1_results = {
        "name": file1.filename,
        "faces": faces1
    }

    image2_results = {
        "name": file2.filename,
        "faces": faces2
    }

    return [image1_results, image2_results]


@app.route('/', methods=['GET', 'POST'])
def upload_images_and_compare():
    if request.method == 'POST':
        files = request.files.getlist("file")

        if len(files) != 2:
            return render_template('index.html')

        # Get the selected Face detection model to use from the form data.
        # "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
        # deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        detection_method = request.form.get("method", "hog")

        results = {}

        # for file in files:
        #     if file.filename == '':
        #         continue
        #
        #     if file and allowed_file(file.filename):
        #         # Save the uploaded image temporarily to display it below the appropriate form
        #         filename = os.path.join("static", file.filename)
        #         file.save(filename)

        # Perform face recognition on the uploaded images
        if files[0] and allowed_file(files[0].filename) and files[1] and allowed_file(files[1].filename):
            results = compare_faces_in_images(file1=files[0], file2=files[1], method=detection_method)

        # Check if there are any results to display
        if len(results) == 0:
            no_results_message = "No results to display. Please upload two valid images for comparison."
            return render_template('results.html', results=[], no_results_message=no_results_message)

        return render_template('results.html', results=results)

    return render_template('index.html')


@app.route('/upload', methods=['GET'])
def upload_new_images():
    return redirect('/')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
