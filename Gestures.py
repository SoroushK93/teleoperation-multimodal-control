'''
This code is essentially defining hand gestures in a separate code so it can be called rather than pasted in.
'''

def gestures(landmarks, label=None):
    gesture = None
    handedness = None
    if label == 'Right':
        handedness = label.lower()
    if label == 'Left':
        handedness = label.lower()
    if all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [8, 12, 16, 20]) and \
                all(landmarks[tip][1] < landmarks[0][1] for tip in [8, 12, 16, 20]):
        gesture = "open"
        # all fingers pointing up and all tips above wrist

    if all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [8, 12]) and \
           all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [16, 20]):
        gesture = "peace"
        # index and middle pointing up, ring and pinky pointing down

    if all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [12]) and \
           all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [8, 16, 20]):
        gesture = "middle"
        # middle pointing up; index, ring and pinky pointing down

    if all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [8, 12, 16]) and \
           all(landmarks[tip][1] > landmarks[tip - 1][1] for tip in [20]):
        gesture = "three"
        # index, middle, ring pointing up; pinky pointing down

    if all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [8, 12]) and \
           all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [16, 20]):
        gesture = "last two"
        # ring and pinky pointing up, index and middle pointing down

    if all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [8, 20]) and \
           all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [12, 16]):
        gesture = "rocknroll"
        # index and pinky pointing up, middle and ring pointing down

    if all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [8, 20]) and \
           all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [12, 16]):
        gesture = "down rocknroll"
        # ring and middle pointing up, index and pinky pointing down

    if all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [8, 12, 16]) and \
           all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [20]):
        gesture = "pinky"
        # pinky pointing up; index, middle and ring pointing down

    if all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [8, 12, 16]) and \
           all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [20]) and all(landmarks[20][1] > landmarks[mark][1] for mark in [0,2,3,6,5,8,9,12,13,16,17]):
        gesture = "pinky down"
        # index, middle and ring pointing up; pinky pointing down; and pinky tip below all other points

    if all(landmarks[4][1] < landmarks[mark][1] for mark in [0,2,3,6,5,8,9,12,13,16,17,20]) and \
        all(landmarks[3][1] < landmarks[mark][1] for mark in [0,2,6,5,8,9,12,13,16,17,20]):
        gesture = "thumbs up"
        # thumb tip and second joint above most other points

    if all(landmarks[4][1] > landmarks[mark][1] for mark in [0,2,3,6,5,8,9,12,13,16,17,20]) and \
        all(landmarks[3][1] > landmarks[mark][1] for mark in [0,2,6,5,8,9,12,13,16,17,20]):
        gesture = "thumbs down"
        # thumb tip and second joint below most other points

    if all(landmarks[8][1] < landmarks[mark][1] for mark in [0,2,3,6,5,9,12,13,16,17,20]) and \
           all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in [12, 16, 20]):
        gesture = "point up"
        # index pointing up, is above most other points, and ring, middle and pinky pointing down

    if all(landmarks[8][1] > landmarks[mark][1] for mark in [0,2,3,5,6,9,12,13,16,17,20]) and \
           all(landmarks[tip][1] < landmarks[tip - 2][1] for tip in [12, 16, 20]):
        gesture = "point down"
        # index pointing down, is below most other points, and ring, middle and pinky pointing up

    if all(landmarks[8][0] < landmarks[mark][0] for mark in [0,2,3,5,6,9,12,13,16,17,20]) and \
           all(landmarks[tip][0] > landmarks[tip - 2][0] for tip in [12, 16, 20]):
        gesture = "point left"
        # index pointing left, is more left than most other points, and ring, middle and pinky pointing right

    if all(landmarks[8][0] > landmarks[mark][0] for mark in [0,2,3,5,6,9,12,13,16,17,20]) and \
           all(landmarks[tip][0] < landmarks[tip - 2][0] for tip in [12, 16, 20]):
        gesture = "point right"
        # index pointing right, is more right than most other points, and ring, middle and pinky pointing left

    if handedness == None:
        return gesture
    return gesture, handedness

# and \
