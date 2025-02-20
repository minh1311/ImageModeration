// /**
//  * @file DualBoxTracker.cpp
//  * @author HuyNQ (huy.nguyen@gpstech.vn)
//  * @brief 
//  * @version 0.1
//  * @date 2024-12-25
//  * 
//  * @copyright Copyright (c) 2024
//  * 
//  */

// #include "DualBoxTracker.h"

// DualTrack::DualTrack(vector<float> tlwh1_, vector<float> tlwh2_, float score1_, float score2_) 
//     : STrack(tlwh1_, score1_), score1(score1_), score2(score2_) {
//     _tlwh1.assign(tlwh1_.begin(), tlwh1_.end());
//     _tlwh2.assign(tlwh2_.begin(), tlwh2_.end());
//     tlwh1.resize(4);
//     tlwh2.resize(4);
//     has_box1 = !tlwh1_.empty();
//     has_box2 = !tlwh2_.empty();
// }

// void DualTrack::update_boxes(const vector<float>& tlwh1, const vector<float>& tlwh2,
//                            float score1_, float score2_, bool has_box1_, bool has_box2_) {
//     if (has_box1_) {
//         this->tlwh1.assign(tlwh1.begin(), tlwh1.end());
//         this->score1 = score1_;
//     }
//     if (has_box2_) {
//         this->tlwh2.assign(tlwh2.begin(), tlwh2.end());
//         this->score2 = score2_;
//     }
//     this->has_box1 = has_box1_;
//     this->has_box2 = has_box2_;
// }

// DualBoxTracker::DualBoxTracker(int frame_rate, int track_buffer) 
//     : tracker1(frame_rate, track_buffer), 
//       tracker2(frame_rate, track_buffer) {
//     frame_id = 0;
//     track_thresh = 0.5;
//     max_time_lost = int(frame_rate / 30.0 * track_buffer);
// }

// vector<DualTrack> DualBoxTracker::update(const vector<DualBoxObject>& objects) {
//     frame_id++;
    
//     // Prepare input for both trackers
//     vector<TrackObject> objects1, objects2;
    
//     for (const auto& obj : objects) {
//         if (obj.has_box1) {
//             TrackObject track_obj1;
//             track_obj1.rect = obj.box1;
//             track_obj1.prob = obj.prob1;
//             track_obj1.label = obj.label;
//             objects1.push_back(track_obj1);
//         }
        
//         if (obj.has_box2) {
//             TrackObject track_obj2;
//             track_obj2.rect = obj.box2;
//             track_obj2.prob = obj.prob2;
//             track_obj2.label = obj.label;
//             objects2.push_back(track_obj2);
//         }
//     }
    
//     // Update individual trackers
//     vector<STrack> tracks1 = tracker1.update(objects1);
//     vector<STrack> tracks2 = tracker2.update(objects2);
    
//     // Match and update dual tracks
//     update_tracks(tracks1, tracks2);
    
//     // Return active tracks
//     vector<DualTrack> output_tracks;
//     for (const auto& track : tracked_tracks) {
//         if (track.is_activated) {
//             output_tracks.push_back(track);
//         }
//     }
    
//     return output_tracks;
// }

// void DualBoxTracker::update_tracks(const vector<STrack>& tracks1, const vector<STrack>& tracks2) {
//     // Match tracks from both trackers based on IDs and spatial relationship
//     match_tracks(tracks1, tracks2);
    
//     // Update lost tracks
//     vector<DualTrack> new_lost_tracks;
//     for (const auto& track : tracked_tracks) {
//         if (!track.is_activated || 
//             (frame_id - track.end_frame() > max_time_lost)) {
//             new_lost_tracks.push_back(track);
//         }
//     }
    
//     // Remove lost tracks
//     for (const auto& lost_track : new_lost_tracks) {
//         lost_tracks.push_back(lost_track);
//     }
    
//     // Remove lost tracks from tracked tracks
//     tracked_tracks.erase(
//         std::remove_if(tracked_tracks.begin(), tracked_tracks.end(),
//             [this](const DualTrack& t) { 
//                 return frame_id - t.end_frame() > max_time_lost; 
//             }
//         ),
//         tracked_tracks.end()
//     );
// }

// void DualBoxTracker::match_tracks(const vector<STrack>& tracks1, const vector<STrack>& tracks2) {
//     // Create map of track IDs to tracks for both trackers
//     unordered_map<int, const STrack*> id_to_track1;
//     unordered_map<int, const STrack*> id_to_track2;
    
//     for (const auto& track : tracks1) {
//         id_to_track1[track.track_id] = &track;
//     }
    
//     for (const auto& track : tracks2) {
//         id_to_track2[track.track_id] = &track;
//     }
    
//     // Update existing tracks and create new ones
//     vector<DualTrack> new_tracks;
    
//     // Update existing tracks
//     for (auto& track : tracked_tracks) {
//         const STrack* track1 = id_to_track1[track.track_id];
//         const STrack* track2 = id_to_track2[track.track_id];
        
//         if (track1 || track2) {
//             // Update existing track
//             vector<float> tlwh1 = track1 ? track1->tlwh : vector<float>();
//             vector<float> tlwh2 = track2 ? track2->tlwh : vector<float>();
//             float score1 = track1 ? track1->score : 0;
//             float score2 = track2 ? track2->score : 0;
            
//             track.update_boxes(tlwh1, tlwh2, score1, score2, track1 != nullptr, track2 != nullptr);
//         }
//     }
    
//     // Create new tracks for unmatched detections
//     set<int> existing_ids;
//     for (const auto& track : tracked_tracks) {
//         existing_ids.insert(track.track_id);
//     }
    
//     for (const auto& track : tracks1) {
//         if (existing_ids.find(track.track_id) == existing_ids.end()) {
//             vector<float> empty_tlwh;
//             DualTrack new_track(track.tlwh, empty_tlwh, track.score, 0);
//             new_track.track_id = track.track_id;
//             new_track.is_activated = track.is_activated;
//             new_track.frame_id = track.frame_id;
//             new_track.start_frame = track.start_frame;
//             new_track.state = track.state;
//             new_tracks.push_back(new_track);
//         }
//     }
    
//     for (const auto& track : tracks2) {
//         if (existing_ids.find(track.track_id) == existing_ids.end()) {
//             vector<float> empty_tlwh;
//             DualTrack new_track(empty_tlwh, track.tlwh, 0, track.score);
//             new_track.track_id = track.track_id;
//             new_track.is_activated = track.is_activated;
//             new_track.frame_id = track.frame_id;
//             new_track.start_frame = track.start_frame;
//             new_track.state = track.state;
//             new_tracks.push_back(new_track);
//         }
//     }
    
//     // Add new tracks to tracked tracks
//     tracked_tracks.insert(tracked_tracks.end(), new_tracks.begin(), new_tracks.end());
// }

// vector<DualTrack> DualBoxTracker::getLostTracks() {
//     return lost_tracks;
// }

// vector<DualTrack> DualBoxTracker::getSTrack() {
//     return lost_tracks;
// }