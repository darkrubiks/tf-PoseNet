import tensorflow as tf
import cv2
import numpy as np

BODY_PARTS = {"nose": 0, "leftEye": 1, "rightEye": 2, "leftEar": 3, "rightEar": 4,
                   "leftShoulder": 5, "rightShoulder": 6, "leftElbow": 7, "rightElbow": 8, "leftWrist": 9,
                   "rightWrist": 10, "leftHip": 11, "rightHip": 12, "leftKnee": 13, "rightKnee": 14,
                   "leftAnkle": 15, "rightAnkle": 16}

BODY_PARTS_COLORS = {"nose": (153,0,0), "leftEye":(153,0,153) , "rightEye":(102,0,153) , "leftEar": (153,0,50), "rightEar": (153,0,102),
                   "leftShoulder": (51,153,0), "rightShoulder": (153,102,0), "leftElbow": (0,153,0), "rightElbow": (153,153,0), "leftWrist": (0,153,51),
                   "rightWrist": (102,153,0), "leftHip":(0,51,153), "rightHip": (0,153,102), "leftKnee": (0,0,153), "rightKnee": (0,153,153),
                   "leftAnkle": (51,0,153), "rightAnkle": (0,102,153)}

POSE_PAIRS = [['nose','leftEye'], ['nose','rightEye'], ['rightEye','rightEar'], ['leftEye','leftEar'],
              ['rightShoulder','rightElbow'], ['leftShoulder','leftElbow'], ['rightElbow','rightWrist'],
              ['leftElbow','leftWrist'], ['rightShoulder','leftShoulder'], ['rightShoulder','rightHip'],
              ['leftShoulder','leftHip'], ['rightHip','rightKnee'], ['leftHip','leftKnee'], ['rightKnee','rightAnkle'],
              ['leftKnee','leftAnkle'],['rightHip','leftHip']]

interpreter = tf.lite.Interpreter('.\\posenet_mobilenet.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

inHeight = input_details[0]['shape'][1]
inWidth = input_details[0]['shape'][2]

cap = cv2.VideoCapture('test6.mp4')

while cv2.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        break
        
    frameHeight = frame.shape[0]
    frameWidth= frame.shape[1]
            
    inFrame = cv2.resize(frame, (inWidth, inHeight))
    inFrame = cv2.cvtColor(inFrame, cv2.COLOR_BGR2RGB)

    inFrame = np.expand_dims(inFrame, axis=0)
    inFrame = (np.float32(inFrame) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], inFrame)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    offset_data = interpreter.get_tensor(output_details[1]['index'])

    heatmaps = np.squeeze(output_data)
    offsets = np.squeeze(offset_data)
    
    points = []
    joint_num = heatmaps.shape[-1]

    for i in range(heatmaps.shape[-1]):
        joint_heatmap = heatmaps[...,i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos / 8 * 257,dtype=np.int32)
        point_y = int(remap_pos[0] + offsets[max_val_pos[0],max_val_pos[1],i])
        point_x = int(remap_pos[1] + offsets[max_val_pos[0],max_val_pos[1],i + joint_num])
        conf = np.max(joint_heatmap)

        x = (frameWidth * point_x) / inWidth
        y = (frameHeight * point_y) / inHeight

        points.append((int(x), int(y)) if conf > 0.3 else None)
        
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        
        colorFrom = BODY_PARTS_COLORS[partFrom]
        colorTo = BODY_PARTS_COLORS[partTo]
        
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], colorFrom, 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, colorFrom, cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, colorTo, cv2.FILLED)

    frame = cv2.resize(frame,(int(frameWidth/1.5),int(frameHeight/1.5)))        
    cv2.imshow('tf-PoseNet', frame)

cap.release()
cv2.destroyAllWindows()                 