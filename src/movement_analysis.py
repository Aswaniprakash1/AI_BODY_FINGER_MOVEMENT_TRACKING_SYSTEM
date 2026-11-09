"""Confidence-filtered geometric posture/action rules."""
import math
from collections import deque, Counter

KEYPOINT_CONFIDENCE_THRESHOLD = 0.25

def _point(person, name, threshold=KEYPOINT_CONFIDENCE_THRESHOLD):
    p = person.get("keypoints", {}).get(name)
    return ((p[0], p[1]) if p and len(p) >= 3
            and all(math.isfinite(value) for value in p[:3]) and p[2] >= threshold else None)

def valid_keypoint_count(person, threshold=KEYPOINT_CONFIDENCE_THRESHOLD):
    return sum(1 for value in person.get("keypoints", {}).values()
               if value and len(value) >= 3 and all(math.isfinite(item) for item in value[:3])
               and value[2] >= threshold)

def _angle(a, b, c):
    if not all((a,b,c)): return None
    v1=(a[0]-b[0],a[1]-b[1]); v2=(c[0]-b[0],c[1]-b[1])
    den=math.hypot(*v1)*math.hypot(*v2)
    return math.degrees(math.acos(max(-1,min(1,(v1[0]*v2[0]+v1[1]*v2[1])/den)))) if den else None

def analyze_posture(person):
    kp=person.get("keypoints", {})
    ls,rs,lh,rh,lk,rk,lw,rw=[_point(person,n) for n in ("left_shoulder","right_shoulder","left_hip","right_hip","left_knee","right_knee","left_wrist","right_wrist")]
    shoulder=( (ls[0]+rs[0])/2,(ls[1]+rs[1])/2 ) if ls and rs else ls or rs
    hip=( (lh[0]+rh[0])/2,(lh[1]+rh[1])/2 ) if lh and rh else lh or rh
    knee=( (lk[0]+rk[0])/2,(lk[1]+rk[1])/2 ) if lk and rk else lk or rk
    knee_angles=[a for a in (_angle(lh,lk, _point(person,"left_ankle")),_angle(rh,rk,_point(person,"right_ankle"))) if a is not None]
    raised=bool(shoulder and ((lw and lw[1]<shoulder[1]-15) or (rw and rw[1]<shoulder[1]-15)))
    if shoulder and hip and knee_angles and sum(knee_angles)/len(knee_angles)<125: posture="Sitting"
    elif shoulder and hip and abs(shoulder[0]-hip[0])>max(35,abs(shoulder[1]-hip[1])*0.8): posture="Bending"
    elif shoulder and hip and knee_angles: posture="Standing"
    elif shoulder and hip: posture="Upright torso"
    elif hip and knee: posture="Lower body visible"
    else: posture="Insufficient landmarks"
    return {"posture":posture,"hand_raised":raised}

class TemporalAnalyzer:
    def __init__(self, window=12): self.history=deque(maxlen=window); self.centers=deque(maxlen=window); self.hip_y=deque(maxlen=window)
    def update(self, people):
        if not people:
            self.centers.clear(); self.hip_y.clear()
            self.history.append("No person detected"); return {"movement":"No person detected","posture":"Unknown","hand_raised":False}
        person=people[0]
        result=analyze_posture(person)
        if valid_keypoint_count(person) < 2:
            self.centers.clear(); self.hip_y.clear()
            self.history.append("Insufficient landmarks")
            return {"movement":"Insufficient landmarks","posture":"Insufficient landmarks","hand_raised":False}
        box=person.get("bbox",(0,0,0,0))
        if len(box) >= 4 and all(math.isfinite(value) for value in box):
            self.centers.append(((box[0]+box[2])/2,(box[1]+box[3])/2))
        hip=_point(person,"left_hip") or _point(person,"right_hip")
        if hip: self.hip_y.append(hip[1])
        dynamic=""
        if len(self.centers)>=5:
            dx=abs(self.centers[-1][0]-self.centers[0][0]); dy=abs(self.centers[-1][1]-self.centers[0][1])
            if dy>45: dynamic="Jumping"
            elif dx>35: dynamic="Running" if dx>100 else "Walking"
        movement=dynamic or ("Hand raised" if result["hand_raised"] else result["posture"])
        self.history.append(movement)
        stable=Counter(self.history).most_common(1)[0][0]
        return {"movement":stable,"posture":result["posture"],"hand_raised":result["hand_raised"]}
