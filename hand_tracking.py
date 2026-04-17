import cv2
import mediapipe as mp
import socket  
import math

# 1. Set up the new Tasks API shortcuts
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 2. Configure options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE, 
    num_hands=1
)

# --- UNITY BROADCAST SETUP (Ignored for now) ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

print("Starting camera... Press 'q' to quit.")
cv2.namedWindow('Modern Tasks API Tracking', cv2.WINDOW_NORMAL)

previous_points = []
smoothing_factor = 0.5

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2
        
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 1)
        cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = landmarker.detect(mp_image)
        
        if result.hand_landmarks and result.hand_world_landmarks:
            
            # --- UNITY BROADCAST LOOP (Commented out for now) ---
            # world_points = result.hand_world_landmarks[0]
            # data_to_send = []
            # for landmark in world_points:
            #     data_to_send.append(landmark.x * 100)
            #     data_to_send.append(-landmark.y * 100)
            #     data_to_send.append(-landmark.z * 100)
            # message = ",".join(map(str, data_to_send))
            # sock.sendto(message.encode(), serverAddressPort)
            # ---------------------------------------------------

            for hand_points in result.hand_landmarks:
                points = []
                is_first_frame = len(previous_points) == 0
                
                for idx, landmark in enumerate(hand_points):
                    raw_cx, raw_cy = int(landmark.x * w), int(landmark.y * h)
                    
                    if is_first_frame:
                        smoothed_cx, smoothed_cy = raw_cx, raw_cy
                        previous_points.append((smoothed_cx, smoothed_cy))
                    else:
                        prev_cx, prev_cy = previous_points[idx]
                        smoothed_cx = int(prev_cx + (raw_cx - prev_cx) * smoothing_factor)
                        smoothed_cy = int(prev_cy + (raw_cy - prev_cy) * smoothing_factor)
                        previous_points[idx] = (smoothed_cx, smoothed_cy)
                    
                    cx, cy = smoothed_cx, smoothed_cy
                    points.append((cx, cy)) 
                    
                    # --- OUR X, Y, Z TRACKING LOGIC ---
                    if idx == 9:
                        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
                        
                        math_x = cx - center_x
                        math_y = center_y - cy
                        
                        wrist_x, wrist_y = points[0] 
                        
                        hand_size_pixels = math.hypot(cx - wrist_x, cy - wrist_y)
                        
                        math_z = int(10000 / hand_size_pixels) if hand_size_pixels > 0 else 0
                        
                        text = f"X: {math_x} | Y: {math_y} | Z: {math_z}"
                        cv2.putText(frame, text, (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                
                # Draw the skeleton connections
                HAND_CONNECTIONS = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
                ]
                
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        else:
            previous_points = []
                    
        cv2.imshow('Modern Tasks API Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()