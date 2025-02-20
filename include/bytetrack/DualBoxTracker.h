// /**
//  * @file DualBoxTracker.h
//  * @author HuyNQ (huy.nguyen@gpstech.vn)
//  * @brief 
//  * @version 0.1
//  * @date 2024-12-25
//  * 
//  * @copyright Copyright (c) 2024
//  * 
//  */

// #pragma once
// #include "BYTETracker.h"
// #include "kalmanFilter.h"

// struct DualBoxObject {
//     cv::Rect_<float> box1;
//     cv::Rect_<float> box2;
//     float prob1;
//     float prob2;
//     int label;
//     bool has_box1;
//     bool has_box2;
// };

// class DualTrack {
// public:
//     DualTrack(const DualBoxObject& obj) {
//         track_id = 0;
//         frame_id = 0;
//         tracklet_len = 0;
//         start_frame = 0;
//         is_activated = false;
//         state = TrackState::New;
        
//         has_box1 = obj.has_box1;
//         has_box2 = obj.has_box2;
        
//         if (has_box1) {
//             tlwh1 = cv_rect_to_tlwh(obj.box1);
//             prob1 = obj.prob1;
//         }
//         if (has_box2) {
//             tlwh2 = cv_rect_to_tlwh(obj.box2);
//             prob2 = obj.prob2;
//         }
        
//         label = obj.label;
//     }
    
//     void activate(byte_kalman::KalmanFilter& kalman_filter, int frame_id) {
//         this->track_id = next_id();
//         this->tracklet_len = 0;
//         this->state = TrackState::Tracked;
        
//         if (has_box1) {
//             vector<float> xyah1 = tlwh_to_xyah(tlwh1);
//             DETECTBOX xyah_box1;
//             for (int i = 0; i < 4; i++) xyah_box1[i] = xyah1[i];
//             auto mc1 = kalman_filter.initiate(xyah_box1);
//             mean1 = mc1.first;
//             covariance1 = mc1.second;
//         }
        
//         if (has_box2) {
//             vector<float> xyah2 = tlwh_to_xyah(tlwh2);
//             DETECTBOX xyah_box2;
//             for (int i = 0; i < 4; i++) xyah_box2[i] = xyah2[i];
//             auto mc2 = kalman_filter.initiate(xyah_box2);
//             mean2 = mc2.first;
//             covariance2 = mc2.second;
//         }
        
//         if (frame_id == 1) {
//             this->is_activated = true;
//         }
//         this->frame_id = frame_id;
//         this->start_frame = frame_id;
//     }
    
//     void predict(byte_kalman::KalmanFilter& kalman_filter) {
//         if (has_box1) {
//             kalman_filter.predict(mean1, covariance1);
//             static_tlwh1();
//         }
//         if (has_box2) {
//             kalman_filter.predict(mean2, covariance2);
//             static_tlwh2();
//         }
//     }
    
//     void update(DualBoxObject& obj, byte_kalman::KalmanFilter& kalman_filter, int frame_id) {
//         this->frame_id = frame_id;
//         this->tracklet_len++;
        
//         // Update box1
//         if (obj.has_box1) {
//             vector<float> new_tlwh1 = cv_rect_to_tlwh(obj.box1);
//             vector<float> xyah1 = tlwh_to_xyah(new_tlwh1);
//             DETECTBOX xyah_box1;
//             for (int i = 0; i < 4; i++) xyah_box1[i] = xyah1[i];
            
//             if (has_box1) {
//                 auto mc1 = kalman_filter.update(mean1, covariance1, xyah_box1);
//                 mean1 = mc1.first;
//                 covariance1 = mc1.second;
//             } else {
//                 auto mc1 = kalman_filter.initiate(xyah_box1);
//                 mean1 = mc1.first;
//                 covariance1 = mc1.second;
//             }
            
//             has_box1 = true;
//             tlwh1 = new_tlwh1;
//             prob1 = obj.prob1;
//             static_tlwh1();
//         }
        
//         // Update box2
//         if (obj.has_box2) {
//             vector<float> new_tlwh2 = cv_rect_to_tlwh(obj.box2);
//             vector<float> xyah2 = tlwh_to_xyah(new_tlwh2);
//             DETECTBOX xyah_box2;
//             for (int i = 0; i < 4; i++) xyah_box2[i] = xyah2[i];
            
//             if (has_box2) {
//                 auto mc2 = kalman_filter.update(mean2, covariance2, xyah_box2);
//                 mean2 = mc2.first;
//                 covariance2 = mc2.second;
//             } else {
//                 auto mc2 = kalman_filter.initiate(xyah_box2);
//                 mean2 = mc2.first;
//                 covariance2 = mc2.second;
//             }
            
//             has_box2 = true;
//             tlwh2 = new_tlwh2;
//             prob2 = obj.prob2;
//             static_tlwh2();
//         }
        
//         label = obj.label;        
//         state = TrackState::Tracked;
//         is_activated = true;
//     }
    
//     vector<float> tlwh_to_xyah(const vector<float>& tlwh) {
//         vector<float> xyah(4);
//         xyah[0] = tlwh[0] + tlwh[2] / 2;  // center x
//         xyah[1] = tlwh[1] + tlwh[3] / 2;  // center y
//         xyah[2] = tlwh[2] / tlwh[3];      // aspect ratio
//         xyah[3] = tlwh[3];                // height
//         return xyah;
//     }
//     void mark_lost(){
//         state = TrackState::Lost;
//     }
//     int end_frame(){
//         return this->frame_id;
//     }
//     void mark_removed(){
//         state = TrackState::Removed;
//     }
    
// private:
//     void static_tlwh1() {
//         if (state == TrackState::New) {
//             return;
//         }
        
//         // Convert Kalman state to tlwh format
//         float x = mean1(0);
//         float y = mean1(1);
//         float aspect_ratio = mean1(2);
//         float height = mean1(3);
        
//         float width = height * aspect_ratio;
//         tlwh1[0] = x - width / 2;  // left
//         tlwh1[1] = y - height / 2; // top
//         tlwh1[2] = width;
//         tlwh1[3] = height;
//     }
    
//     void static_tlwh2() {
//         if (state == TrackState::New) {
//             return;
//         }
        
//         // Convert Kalman state to tlwh format
//         float x = mean2(0);
//         float y = mean2(1);
//         float aspect_ratio = mean2(2);
//         float height = mean2(3);
        
//         float width = height * aspect_ratio;
//         tlwh2[0] = x - width / 2;  // left
//         tlwh2[1] = y - height / 2; // top
//         tlwh2[2] = width;
//         tlwh2[3] = height;
//     }
    
//     vector<float> cv_rect_to_tlwh(const cv::Rect_<float>& rect) {
//         vector<float> tlwh(4);
//         tlwh[0] = rect.x;
//         tlwh[1] = rect.y;
//         tlwh[2] = rect.width;
//         tlwh[3] = rect.height;
//         return tlwh;
//     }
    
//     int next_id() {
//         static int _count = 0;
//         _count++;
//         return _count;
//     }

// public:
//     // Track information
//     int track_id;
//     int frame_id;
//     int start_frame;
//     int tracklet_len;
//     int state;
//     bool is_activated;
    
//     // Box 1 information
//     vector<float> tlwh1;
//     float prob1;
//     bool has_box1;
//     KAL_MEAN mean1;
//     KAL_COVA covariance1;
    
//     // Box 2 information
//     vector<float> tlwh2;
//     float prob2;
//     bool has_box2;
//     KAL_MEAN mean2;
//     KAL_COVA covariance2;
//     bool bLost;
//     int label;
// };

// class DualBoxTracker {
// public:
//     DualBoxTracker(int frame_rate = 30, int track_buffer = 30) {
//         track_thresh = 0.5;
//         frame_id = 0;
//         max_time_lost = int(frame_rate / 30.0 * track_buffer);
//     }
    
//     vector<DualTrack> update(vector<DualBoxObject>& objects) {
//         frame_id++;
        
//         vector<DualTrack> activated_tracks;
//         vector<DualTrack> refind_tracks;
//         vector<DualTrack> lost_tracks;
//         vector<DualTrack> removed_tracks;
        
//         // Step 1: Predict states using Kalman filter
//         for (auto& track : tracked_stracks) {
//             track.predict(kalman_filter);
//         }
        
//         // Step 2: Process new detections
//         vector<DualTrack> detections;
//         for (const auto& obj : objects) {
//             if ((obj.has_box1 && obj.prob1 >= track_thresh) || 
//                 (obj.has_box2 && obj.prob2 >= track_thresh)) {
//                 detections.emplace_back(obj);
//             }
//         }
        
//         // Get unconfirmed & tracked tracks
//         vector<DualTrack*> unconfirmed_tracks;
//         vector<DualTrack*> tracked_tracks;
//         for (auto& track : tracked_stracks) {
//             if (!track.is_activated) {
//                 unconfirmed_tracks.push_back(&track);
//             } else {
//                 tracked_tracks.push_back(&track);
//             }
//         }
        
//         // Step 3: Match with existing tracks
//         match_tracks(detections, tracked_tracks, activated_tracks, refind_tracks, lost_tracks);
        
//         // Step 4: Match with unconfirmed tracks
//         match_unconfirmed(detections, unconfirmed_tracks, activated_tracks, removed_tracks);
        
//         // Step 5: Initialize new tracks
//         for (auto& detection : detections) {
//             if (!detection.is_activated) {
//                 detection.activate(kalman_filter, frame_id);
//                 activated_tracks.push_back(detection);
//             }
//         }
        
//         // Step 6: Update state
//         for (auto& track : lost_stracks) {
//             if (frame_id - track.end_frame() > max_time_lost) {
//                 track.mark_removed();
//                 removed_tracks.push_back(track);
//             }
//         }
        
//         // Update tracked_stracks
//         tracked_stracks.clear();
//         for (auto& track : activated_tracks) {
//             tracked_stracks.push_back(track);
//         }
//         for (auto& track : refind_tracks) {
//             tracked_stracks.push_back(track);
//         }
        
//         // Update lost_stracks
//         lost_stracks.insert(lost_stracks.end(), lost_tracks.begin(), lost_tracks.end());
        
//         // Get output tracks
//         vector<DualTrack> output_stracks;
//         for (auto& track : tracked_stracks) {
//             if (track.is_activated) {
//                 output_stracks.push_back(track);
//             }
//         }
        
//         return output_stracks;
//     }
    
//     vector<DualTrack> getLostTracks() {
//         vector<DualTrack> output_tracks;
//         for (auto& track : lost_stracks) {
//             if (track.is_activated) {
//                 output_tracks.push_back(track);
//             }
//         }
//         return output_tracks;
//     }
//     vector<DualTrack> getSTrack(){
//         std::vector<DualTrack> output_stracks;

//         for (auto& strack : this->lost_stracks)
//         {
//             if ((strack.is_activated) && (strack.state == TrackState::Lost))
//             {
//                 strack.bLost= true;
//                 output_stracks.push_back(strack);
//             }
//         }

//         for (auto& strack : this->tracked_stracks)
//         {
//             if (strack.is_activated == true)
//             {
//                 strack.bLost= false;
//                 output_stracks.push_back(strack);
//             }
//         }

//         return output_stracks;
//     }
    
// private:
//     void match_tracks(vector<DualTrack>& detections,
//                  vector<DualTrack*>& tracked_tracks,
//                  vector<DualTrack>& activated_tracks,
//                  vector<DualTrack>& refind_tracks,
//                  vector<DualTrack>& lost_tracks) {
    
//         if (tracked_tracks.empty()) {
//             return;
//         }
        
//         // Calculate IoU matrix
//         vector<vector<float>> iou_matrix;
//         for (const auto& detection : detections) {
//             vector<float> ious;
//             for (const auto& track : tracked_tracks) {
//                 float iou = 0;
//                 int match_count = 0;
                
//                 if (detection.has_box1 && track->has_box1) {
//                     iou += calculate_iou(detection.tlwh1, track->tlwh1);
//                     match_count++;
//                 }
//                 if (detection.has_box2 && track->has_box2) {
//                     iou += calculate_iou(detection.tlwh2, track->tlwh2);
//                     match_count++;
//                 }
                
//                 if (match_count > 0) {
//                     iou /= match_count;  // Average IoU
//                 }
                
//                 ious.push_back(iou);
//             }
//             iou_matrix.push_back(ious);
//         }
        
//         // Match using Hungarian algorithm
//         vector<int> detection_matches(detections.size(), -1);
//         vector<int> track_matches(tracked_tracks.size(), -1);
        
//         for (int i = 0; i < iou_matrix.size(); i++) {
//             int best_track = -1;
//             float best_iou = track_thresh;
            
//             for (int j = 0; j < iou_matrix[i].size(); j++) {
//                 if (track_matches[j] == -1 && iou_matrix[i][j] > best_iou) {
//                     best_iou = iou_matrix[i][j];
//                     best_track = j;
//                 }
//             }
            
//             if (best_track != -1) {
//                 detection_matches[i] = best_track;
//                 track_matches[best_track] = i;
//             }
//         }
        
//         // Update matched tracks
//         for (int i = 0; i < detection_matches.size(); i++) {
//             if (detection_matches[i] != -1) {
//                 auto& track = tracked_tracks[detection_matches[i]];
//                 // Create DualBoxObject from DualTrack
//                 DualBoxObject obj;
                
//                 // Convert tlwh format to cv::Rect
//                 if (detections[i].has_box1) {
//                     obj.box1 = cv::Rect_<float>(
//                         detections[i].tlwh1[0],
//                         detections[i].tlwh1[1],
//                         detections[i].tlwh1[2],
//                         detections[i].tlwh1[3]
//                     );
//                     obj.prob1 = detections[i].prob1;
//                 }
//                 if (detections[i].has_box2) {
//                     obj.box2 = cv::Rect_<float>(
//                         detections[i].tlwh2[0],
//                         detections[i].tlwh2[1],
//                         detections[i].tlwh2[2],
//                         detections[i].tlwh2[3]
//                     );
//                     obj.prob2 = detections[i].prob2;
//                 }
                
//                 obj.has_box1 = detections[i].has_box1;
//                 obj.has_box2 = detections[i].has_box2;
//                 obj.label = detections[i].label;
                
//                 track->update(obj, kalman_filter, frame_id);
//                 activated_tracks.push_back(*track);
//             }
//         }
        
//         // Handle unmatched tracks
//         for (int i = 0; i < track_matches.size(); i++) {
//             if (track_matches[i] == -1) {
//                 tracked_tracks[i]->mark_lost();
//                 lost_tracks.push_back(*tracked_tracks[i]);
//             }
//         }
//     }
    
//     void match_unconfirmed(vector<DualTrack>& detections,
//                           vector<DualTrack*>& unconfirmed_tracks,
//                           vector<DualTrack>& activated_tracks,
//                           vector<DualTrack>& removed_tracks) {
//         // Similar to match_tracks but with higher threshold
//         // Implementation details omitted for brevity
//     }
    
//     float calculate_iou(const vector<float>& box1, const vector<float>& box2) {
//         float x1 = max(box1[0], box2[0]);
//         float y1 = max(box1[1], box2[1]);
//         float x2 = min(box1[0] + box1[2], box2[0] + box2[2]);
//         float y2 = min(box1[1] + box1[3], box2[1] + box2[3]);
        
//         float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
//         float area1 = box1[2] * box1[3];
//         float area2 = box2[2] * box2[3];
//         float union_area = area1 + area2 - intersection;
        
//         return intersection / (union_area + 1e-6f);
//     }
    
// private:
//     vector<DualTrack> tracked_stracks;
//     vector<DualTrack> lost_stracks;
//     vector<DualTrack> removed_stracks;
    
//     byte_kalman::KalmanFilter kalman_filter;
//     int frame_id;
//     float track_thresh;
//     int max_time_lost;
// };