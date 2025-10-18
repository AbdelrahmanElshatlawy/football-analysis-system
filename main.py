import os
import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from itertools import count
from ultralytics import YOLO
from IPython.display import Video, Image, display
from team_assigner import TeamAssigner  # your package/module

# ---------------------- model ----------------------
model = YOLO(r"models\fine_tuning_yolo11x\best.pt")

# ---------------------- 1-based display IDs (unchanged) ----------------------
_id_gen = count(1)   # generates 1, 2, 3, ...
_id_map = {}         # {tracker_tid: display_id}
def remap(tid):
    """Stable 1-based ID for display, mapped from the tracker's internal tid."""
    tid = int(tid)
    if tid not in _id_map:
        _id_map[tid] = next(_id_gen)
    return _id_map[tid]

# ---------------------- TeamAssigner wiring ----------------------
team_assigner = TeamAssigner()
team_colors_assigned = False
PLAYER_CLASS_ID = 2       # player
GOALKEEPER_CLASS_ID = 1   # goalkeeper
ADDING_TRAJECTORIES = False  # set to True to enable trajectory drawing

# ---------------------- paths ----------------------
video_path = r"C:\Users\aelsh\Downloads\08fd33_4.mp4"
output_path = rf'E:\Foodball_Analysis_System\outputs\tracked_football_video_1, has trajectories: {ADDING_TRAJECTORIES}.mp4'

# make sure output folder exists
pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# ---------------------- io ----------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# ---------------------- tracking ----------------------
results = model.track(
    source=video_path,
    persist=True,
    stream=True,
    classes=[0, 1, 2, 3],  # 0=ball, 1=goalkeeper, 2=player, 3=referee
    tracker="botsort.yaml"
)

# default colors for non-players / fallback
CLASS_COLORS = {
    0: (0, 0, 255),   # ball
    1: (0, 255, 0),   # goalkeeper (will be overridden by team color)
    2: (255, 0, 0),   # player (will be overridden by team color)
    3: (0, 255, 255)  # referee
}

if ADDING_TRAJECTORIES:
    trajectories = defaultdict(list)  # {tracker_tid: [(x1, y1), (x2, y2), ...]}

for res in results:
    frame = res.orig_img.copy()
    if res.boxes.id is None:
        out.write(frame)
        continue

    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    ids   = res.boxes.id.cpu().numpy().astype(int)

    # pull class ids, confidences, and names
    clses = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()   # required since you zip with confs
    names = res.names                      # {class_id: class_name}
    names[1] = 'player'  # keep/remove as you prefer; not related to coloring

    # --------- Fit team colors once when we have ≥2 player detections ----------
    if not team_colors_assigned:
        player_detections = {}
        for (x1, y1, x2, y2), c in zip(boxes, clses):
            if int(c) == PLAYER_CLASS_ID:
                player_detections[len(player_detections)] = {"bbox": (x1, y1, x2, y2)}
        if len(player_detections) >= 2:
            team_assigner.assign_team_color(frame, player_detections)
            team_colors_assigned = True
    # ---------------------------------------------------------------------------

    for (x1, y1, x2, y2), tid, c, conf in zip(boxes, ids, clses, confs):

        # ---------------------------- Draw bounding box + label -----------------------------
        # Use team color for BOTH goalkeepers and players
        if int(c) in (PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID) and team_colors_assigned:
            try:
                team_id = team_assigner.get_player_team(frame, (x1, y1, x2, y2), int(tid))
                team_color = team_assigner.team_colors.get(team_id, (255, 255, 255))
                color = tuple(int(max(0, min(255, v))) for v in team_color)
            except Exception:
                color = CLASS_COLORS.get(int(c), (200, 200, 200))
        else:
            color = CLASS_COLORS.get(int(c), (200, 200, 200))
        
        # thin box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # label (two lines): class name + 1-based ID
        disp_id  = remap(tid)
        cls_name = names.get(int(c), str(c))
        id_text  = f"ID:{disp_id}"

        (tw1, th1), base1 = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        (tw2, th2), base2 = cv2.getTextSize(id_text,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tw = max(tw1, tw2)
        line_gap = 5
        th_total = th1 + line_gap + th2
        y_text = max(y1 - 8, th_total + 4)

        left   = x1
        top    = y_text - th_total - 4
        right  = x1 + tw + 6
        bottom = y_text + base2 - 2
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)

        # white text for classes 1 & 2, black otherwise
        text_color = (0, 0, 0)

        # First line: class name
        cv2.putText(frame, cls_name, (x1 + 2, y_text - th2 - 2 - line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        # Second line: ID
        cv2.putText(frame, id_text, (x1 + 2, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        # ------------------------------------------------------------------------------------



        # -------------------------- Add to trajectory (if enabled) --------------------------
        if ADDING_TRAJECTORIES and int(c) in (PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trajectories[tid].append((cx, cy))

            if len(trajectories[tid]) >= 2:
                for j in range(1, len(trajectories[tid])):
                    cv2.line(frame, trajectories[tid][j - 1], trajectories[tid][j], color, 2)
        # ------------------------------------------------------------------------------------


    out.write(frame)

cap.release()
out.release()
print(f"✅ Tracking video saved → {output_path}")
