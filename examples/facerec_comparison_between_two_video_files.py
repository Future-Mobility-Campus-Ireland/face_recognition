import face_recognition
import cv2
import time
import argparse
import signal
import sys

# This is a demo of running face anonymization verification between two video file and writes results to console output.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

def destroy() -> None:
    sys.exit()


def parse_args() -> argparse.Namespace:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-v1', '--video1', help='select an original video path name', dest='video1_path')
    program.add_argument('-v2', '--video2', help='select an anonymized video path name', dest='video2_path')

    args = program.parse_args()
    return args


def check_args(args: argparse.Namespace) -> bool:
    if not (args.video1_path and args.video2_path):
        print('video1 or video2 paths are missing. Please provide two video file paths by typing -v1 '
              'path_to_video_1 -v2 path_to_video_2')
        return False

    return True


def pre_check() -> bool:
    if sys.version_info < (2, 7):
        print(f'Python version {sys.version.split()[0]} is not supported - please upgrade to 2.7 or higher.')
        return False
    elif sys.version_info < (3, 3):
        print(f'Python version {sys.version.split()[0]} is not supported - please upgrade to 3.3 or higher.')
        return False
    return True


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


def process_video_files(video_file1, video_file2):
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
    while True:
        # Grab a single frame of video
        ret1, frame1 = input_video1.read()
        ret2, frame2 = input_video2.read()
        frame_number += 1

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
                print("There is a match at frame number {}, timestamp {} [ms], on frame locations: {}".format(
                    frame_number,
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


def run() -> None:
    # start measuring exe time
    start_time = time.time()

    if not pre_check():
        return
    args = parse_args()
    args_ok = check_args(args)
    if not args_ok:
        return
    else:
        video_file1 = args.video1_path
        video_file2 = args.video2_path

        process_video_files(video_file1, video_file2)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")


if __name__ == "__main__":
    run()
