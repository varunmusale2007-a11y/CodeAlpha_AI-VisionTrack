from typing import List, Dict, Any, Tuple
import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class STrack:
    """
    Single Track representation for ByteTrack.
    Maintains position, velocity estimation, identity, and lifetime.
    """
    _count = 0

    def __init__(self, bbox: List[float], score: float, class_name: str, class_id: int):
        STrack._count += 1
        self.track_id = STrack._count
        self.bbox = [float(c) for c in bbox]
        self.score = float(score)
        self.class_name = class_name
        self.class_id = class_id
        
        # Velocity vector [vx1, vy1, vx2, vy2] for motion prediction
        self.velocity = [0.0, 0.0, 0.0, 0.0]
        self.time_since_update = 0
        self.age = 0
        self.hits = 1
        self.is_activated = True

    def predict(self):
        """Predict the next position based on velocity and momentum."""
        self.age += 1
        self.time_since_update += 1
        
        # Apply motion projection with friction dampening (0.85)
        self.bbox[0] += self.velocity[0] * 0.85
        self.bbox[1] += self.velocity[1] * 0.85
        self.bbox[2] += self.velocity[2] * 0.85
        self.bbox[3] += self.velocity[3] * 0.85

    def update(self, new_bbox: List[float], new_score: float, new_class: str, new_class_id: int):
        """Update track with a newly matched detection."""
        # Calculate velocity delta
        self.velocity = [
            new_bbox[0] - self.bbox[0],
            new_bbox[1] - self.bbox[1],
            new_bbox[2] - self.bbox[2],
            new_bbox[3] - self.bbox[3]
        ]
        self.bbox = [float(c) for c in new_bbox]
        self.score = float(new_score)
        self.class_name = new_class
        self.class_id = new_class_id
        self.time_since_update = 0
        self.hits += 1

    @classmethod
    def reset_counter(cls):
        cls._count = 0


class ByteTrackTracker:
    """
    ByteTrack Object Tracker.
    Pure tracking algorithm with zero additional deep learning or YOLO overhead.
    Receives detection outputs from ObjectDetector and associates them across frames.
    """
    def __init__(self, high_thresh: float = 0.5, match_thresh: float = 0.4, max_time_lost: int = 30):
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.frame_id = 0
        print("[Tracker] ByteTrack tracking engine initialized (Pure tracking mode).")

    def _associate_detections_to_tracks(
        self,
        tracks: List[STrack],
        detections: List[Dict[str, Any]],
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Bipartite matching between tracks and candidate detections using IoU matrix.
        Returns: (matched_indices, unmatched_track_indices, unmatched_detection_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build IoU cost matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                # Penalize mismatched classes slightly to preserve semantic track consistency
                base_iou = compute_iou(track.bbox, det["bbox"])
                if track.class_id == det["class_id"]:
                    iou_matrix[t_idx, d_idx] = base_iou
                else:
                    iou_matrix[t_idx, d_idx] = base_iou * 0.7

        # Greedy maximum-IoU matching
        matched_tracks = set()
        matched_dets = set()
        matches = []

        # Flatten and sort indices by highest IoU descending
        indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        for t_idx, d_idx in indices:
            if iou_matrix[t_idx, d_idx] < iou_threshold:
                break
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            matches.append((t_idx, d_idx))

        unmatched_tracks = [t for t in range(len(tracks)) if t not in matched_tracks]
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes detections from the single YOLO detector through the ByteTrack algorithm.
        Maintains persistent tracking IDs across sequential frames.
        
        Args:
            detections: List of dicts with keys 'bbox' [x1, y1, x2, y2], 'confidence', 'class', 'class_id'
            
        Returns:
            List of tracked dicts with 'class', 'confidence', 'track_id', and 'bbox'.
        """
        self.frame_id += 1
        
        # 1. Predict position of existing active and lost tracks
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        # 2. Partition detections into high-score and low-score (Core ByteTrack logic)
        dets_high = []
        dets_low = []
        for det in detections:
            if det["confidence"] >= self.high_thresh:
                dets_high.append(det)
            else:
                dets_low.append(det)

        # 3. First Association: Match high-confidence detections with active + lost tracks
        candidates_pool = self.tracked_tracks + self.lost_tracks
        matches_1, unmatched_tracks_1, unmatched_dets_1 = self._associate_detections_to_tracks(
            candidates_pool, dets_high, iou_threshold=self.match_thresh
        )

        updated_tracks: List[STrack] = []
        for t_idx, d_idx in matches_1:
            track = candidates_pool[t_idx]
            det = dets_high[d_idx]
            track.update(det["bbox"], det["confidence"], det["class"], det["class_id"])
            updated_tracks.append(track)

        # 4. Second Association: Match remaining unmatched tracks with low-confidence detections
        remaining_tracks = [candidates_pool[t_idx] for t_idx in unmatched_tracks_1 if candidates_pool[t_idx].time_since_update <= self.max_time_lost]
        matches_2, unmatched_tracks_2, _ = self._associate_detections_to_tracks(
            remaining_tracks, dets_low, iou_threshold=self.match_thresh * 0.8
        )

        for t_idx, d_idx in matches_2:
            track = remaining_tracks[t_idx]
            det = dets_low[d_idx]
            track.update(det["bbox"], det["confidence"], det["class"], det["class_id"])
            updated_tracks.append(track)

        # 5. Initialize new tracks for unmatched high-confidence detections
        for d_idx in unmatched_dets_1:
            det = dets_high[d_idx]
            new_track = STrack(det["bbox"], det["confidence"], det["class"], det["class_id"])
            updated_tracks.append(new_track)

        # 6. Update internal active and lost track lists
        self.tracked_tracks = [t for t in updated_tracks if t.time_since_update == 0]
        
        # Lost tracks: Tracks that were not matched this frame but have not yet timed out
        new_lost = [remaining_tracks[t_idx] for t_idx in unmatched_tracks_2 if remaining_tracks[t_idx].time_since_update <= self.max_time_lost]
        self.lost_tracks = [t for t in new_lost if t not in self.tracked_tracks]

        # 7. Format output
        results: List[Dict[str, Any]] = []
        for track in self.tracked_tracks:
            x1, y1, x2, y2 = [round(float(c), 2) for c in track.bbox]
            results.append({
                "class": track.class_name,
                "confidence": round(float(track.score), 2),
                "track_id": track.track_id,
                "bbox": [x1, y1, x2, y2]
            })

        return results

    def reset(self):
        """Resets tracking state and clears all active track IDs."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.frame_id = 0
        STrack.reset_counter()
        print("[Tracker] ByteTrack tracking state reset.")
