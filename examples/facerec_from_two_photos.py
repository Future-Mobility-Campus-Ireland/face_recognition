import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# This is a demo of running face recognition between two images.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Initialize some variables
face_names = []

# Load original image
original_image = face_recognition.load_image_file("person1.png")
# Find all the faces and face encodings in the original image
face_locations = face_recognition.face_locations(original_image)
original_face_encodings = face_recognition.face_encodings(original_image, face_locations)

# Load anonymized image
anony_image = face_recognition.load_image_file("person1_anon.png")
# Find anonymized face encodings (Assuming the same face locations)
anony_face_encodings = face_recognition.face_encodings(anony_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(anony_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Label the results
# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, original_face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(anony_face_encodings, face_encoding, tolerance=0.50)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(original_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        #name = known_face_names[best_match_index]
        name = 'the same person recognised'

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

