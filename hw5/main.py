import cv2
import numpy as np

POINTS_TO_TRACK = 100
COLORS = np.random.randint(low=0, high=255, size=(POINTS_TO_TRACK, 3))

INPUT_VIDEO_FILENAME = "video.mpg"
HARRIS_OUTPUT_FILENAME = "harris"
FAST_OUTPUT_FILENAME = "fast"


def open_video(input_video, output_video):
    video = cv2.VideoCapture(input_video)
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return frames, video, cv2.VideoWriter("%s.mpg" % output_video, cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), fps, (width, height))


# def draw_features(frame, output_filename, features):
#     frame = frame.copy()
#     for i, feature in enumerate(features):
#         x, y = feature[0]
#         cv2.circle(frame, (x, y), 4, COLORS[i].tolist(), -1)
#     cv2.imwrite("poi_%s.png" % output_filename, frame)


def harris_detector(frame):
    return cv2.goodFeaturesToTrack(image=frame, maxCorners=POINTS_TO_TRACK, qualityLevel=0.15, minDistance=7,
                                   blockSize=7,
                                   useHarrisDetector=True)


def fast_detector(frame):
    fast = cv2.FastFeatureDetector().detect(frame, None)
    fast = list(sorted(fast, key=lambda x: x.response, reverse=True))
    fast = fast[:POINTS_TO_TRACK]
    fast = [[kp.pt] for kp in fast]
    fast = np.array(fast, np.float32)
    return fast


def optical_flow(output_filename, feature_detector):
    # load video
    number_of_frames, source_video, result_video_writer = open_video(INPUT_VIDEO_FILENAME, output_filename)

    # Take first frame and find corners in it
    frame = source_video.read()[1]
    frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features1 = feature_detector(frame_gray1)
    # draw_features(frame, output_filename, features1)

    lukas_kanade = {'winSize': (15, 15),
                    'maxLevel': 2,
                    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    flow = np.zeros_like(frame)
    result_video_writer.write(frame)

    frame_counter = 1
    while True:
        ok, frame = source_video.read()
        if not ok:
            break
        print 'processing %s/%s frame...' % (frame_counter, number_of_frames)
        frame_counter += 1
        frame_gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        features2, status, err = cv2.calcOpticalFlowPyrLK(frame_gray1, frame_gray2, features1, None, **lukas_kanade)

        # Select good points
        good_new = features2[status == 1]
        good_old = features1[status == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(flow, (a, b), (c, d), COLORS[i].tolist(), 2)
            cv2.circle(frame, (a, b), 4, COLORS[i].tolist(), -1)

        result_video_writer.write(cv2.add(frame, flow))
        frame_gray1 = frame_gray2.copy()
        features1 = good_new.reshape(-1, 1, 2)

    result_video_writer.release()
    source_video.release()


if __name__ == "__main__":
    optical_flow(HARRIS_OUTPUT_FILENAME, harris_detector)
    optical_flow(FAST_OUTPUT_FILENAME, fast_detector)