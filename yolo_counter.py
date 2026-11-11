import os
import cv2
from sort import *

prob_min = 0.5 #min confidence for detection
nms_thresh = 0.3 #overlap threshold for removing dupes
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
 
 
def parse_args(): #reads and validates arguments, exits on bad inputs
    parser = argparse.ArgumentParser(
        description="Detect people in a video with YOLOv3, track them with SORT, and count the unique people seen.") 
    parser.add_argument('--weights', required=True, help='Path to YOLO weights (yolov3.weights)') #defines flag, required=True makes it exit if missing, same for cfg, labels, and video
    parser.add_argument('--cfg', required=True, help='Path to YOLO config (yolov3.cfg)')
    parser.add_argument('--labels', required=True, help='Path to class labels (coco.names)')
    parser.add_argument('--video', required=True, help='Path to input video (.mp4)')
    parser.add_argument('--output', default=None, help='Output video path (default: <input dir>/<name>-result.mp4, with a trailing "-raw" removed from the name)') 
    args = parser.parse_args()
 
    for name in ('weights', 'cfg', 'labels', 'video'): #fetches the args, and if it doesn't exist prints a message and exits 
        path = getattr(args, name)
        if not os.path.isfile(path):
            parser.error(f"{name}: file not found: {path}")
    return args
 
 
def default_output_path(video_path):
    folder, fname = os.path.split(video_path) #builds mot_vid/MOTS20-09-result.mp4 from input path It splits path, strips extension, removes the -raw if it's there, and reassembles
    base = os.path.splitext(fname)[0]
    if base.endswith('-raw'):
        base = base[:-len('-raw')]
    return os.path.join(folder, base + '-result.mp4')
 
 
def detect_people(network, frame, person_idx):
    h, w = frame.shape[:2] #gets frame height+width)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False) #prepares frame for network, scaling from 0-1, and resizing to 416x416
    network.setInput(blob) #load blob
    output = network.forward(yolo_layers) #runs inference and returns output arrays
 
    boxes, confidences, class_ids = [], [], []
    for result in output: #loops over each detection from the 3 yolo layers
        for detection in result:
            scores = detection[5:] #takes the 80 scores, finds best class, and gets confidence
            class_current = np.argmax(scores)
            confidence = float(scores[class_current])
            
            if confidence > prob_min: #filter for people only, or if confidence below 0.5
                xc, yc, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype('int') #scales normalized box to pixels, then converts to top-left format
                boxes.append([int(xc - bw / 2), int(yc - bh / 2), int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_current)
 
    dets = [] 
    if len(boxes) > 0:
        keep = cv2.dnn.NMSBoxes(boxes, confidences, prob_min, nms_thresh) #remove overlapping duplicate boxes
        for i in np.array(keep).flatten():
            if class_ids[i] == person_idx:
                x, y, bw, bh = boxes[i]
                dets.append([x, y, x + bw, y + bh, confidences[i]]) #kept boxes are converted to this format
 
    return np.array(dets, dtype=float) if dets else np.empty((0, 5)) #returns detections as array, if none it return empty array
 
 
def annotate(frame, tracks, total_count):
    for x1, y1, x2, y2, track_id in tracks: 
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id) #each sort output row is a box+ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #draws a box and writes ID above it
        cv2.putText(frame, f"ID {track_id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
    cv2.putText(frame, f"Total people counted: {total_count}", (15, 35), #draws the 2 counters in red at top left, the total and have many people were tracked
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, f"Currently tracked: {len(tracks)}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame
 
 
def main():
    args = parse_args() #reads coco.names into list, checks if person exists, and gets index
    with open(args.labels) as f:
        labels = [line.strip() for line in f]
    if 'person' not in labels: #nice exit if person doesn't exist
        print("Error: 'person' class not found in labels file.")
        return 1
    person_idx = labels.index('person')
 
    network = cv2.dnn.readNetFromDarknet(args.cfg, args.weights) #loads yolov3 once before loop
 
    cap = cv2.VideoCapture(args.video) #opens video and exits if not readable
    if not cap.isOpened():
        print(f"Error: could not open video {args.video}")
        return 1
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 30 #reads input frame rate and size for input to match
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    out_path = args.output or default_output_path(args.video) #picks output path and makes folder if it doesn't exist
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) #readies videowriter
 
    tracker = Sort(max_age=0, min_hits=1)
    seen_ids = set() 
    frame_num = 0 
 
    while True: #reads one frame at a time, ok becomes false once video is done, stopping loop
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1
        start = time.time()
 
        dets = detect_people(network, frame, person_idx) #runs yolo to get pepole, and gives to sort, which returns one row per confirmed track
        tracks = tracker.update(dets) #called every frame so filters stay in sync.
 
        for t in tracks: #t[4] is ID column, every ID seen goes into set
            seen_ids.add(int(t[4]))
 
        frame = annotate(frame, tracks, len(seen_ids))#draws overlay writes frame to output video, and prints frame number with processing in seconds
        writer.write(frame)
 
        print(f"Frame {frame_num}: {time.time() - start:.4f} s")
 
    cap.release() #closes input and output
    writer.release()
 
    print(f"\nTotal number of people counted: {len(seen_ids)}") #prints final summary and returns
    print(f"Output video saved to: {out_path}")
    
    return 0
 
 
if __name__ == '__main__': #runs main when file is executed directly.
    raise SystemExit(main())

