import face_recognition
import cv2
import time


# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

def compare_frames(frame1, frame2):
    # Convert frames to face_recognition compatible images (numpy arrays)
    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Perform anonymization verification similar to compare_faces_in_images
    results = compare_faces_in_images(image1, image2)

    return results


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


def compare_faces_in_images(image1, image2, method="hog"):
    face_locations1 = face_recognition.face_locations(image1, model=method)
    face_encodings1 = face_recognition.face_encodings(image1, face_locations1)

    face_locations2 = face_recognition.face_locations(image2, model=method)
    face_encodings2 = face_recognition.face_encodings(image2, face_locations2)

    faces = compare_faces(face_encodings1, face_locations1, face_encodings2, face_locations2)

    return faces


# start measuring exe time
start_time = time.time()

video_file1 = "original_fmci_video_1280.mp4"
video_file2 = "anony_fmci_video_1280.mp4"

# Open the input video files
input_video1 = cv2.VideoCapture(video_file1)
print("Opening and starting processing {}".format(video_file1))
input_video2 = cv2.VideoCapture(video_file2)
print("Opening and starting processing {}".format(video_file2))

length1 = int(input_video1.get(cv2.CAP_PROP_FRAME_COUNT))
length2 = int(input_video2.get(cv2.CAP_PROP_FRAME_COUNT))

fps1 = input_video1.get(cv2.CAP_PROP_FPS)
fps2 = input_video2.get(cv2.CAP_PROP_FPS)

width1 = int(input_video1.get(cv2.CAP_PROP_FRAME_WIDTH))
width2 = int(input_video1.get(cv2.CAP_PROP_FRAME_WIDTH))

height1 = int(input_video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
height2 = int(input_video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

if length1 != length2:
    print("Two videos doesn't have the same length")
    exit()

# Initialize some variables
frame_number = 0
desired_frame = 429
end_frame = 529
while True:
    # Grab a single frame of video
    ret1, frame1 = input_video1.read()
    ret2, frame2 = input_video2.read()
    frame_number += 1

    # Speed up the proces to the first matching frame 429
    if frame_number < desired_frame:
        continue

    if frame_number > end_frame:
        break

    # Quit when the input video file ends
    if not ret1:
        break

    result = compare_frames(frame1, frame2)

    if any(item.get('is_match', False) for item in result):
        timestamp = input_video1.get(cv2.CAP_PROP_POS_MSEC)
        matches = [item for item in result if item.get('is_match', False)]

        if matches:
            match_locations = [str(match.get('location', 'N/A')) for match in matches]
            match_info = ", ".join(match_locations)
            print("There is a match at frame number {}, timestamp {} [ms], on frame locations: {}".format(frame_number,
                                                                                                          timestamp,
                                                                                                          match_info))
        else:
            print("There is a match at frame number {}, timestamp {} [ms], but no matching locations.".format(
                frame_number, timestamp))
    else:
        print("No match found at frame number {}.".format(frame_number))

# All done!
input_video1.release()
input_video2.release()
cv2.destroyAllWindows()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
