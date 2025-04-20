"""
Unit tests for btrack integration with sample data.
"""

import unittest
import matplotlib.pyplot as plt
import os
import btrack
from btrack.constants import BayesianUpdates

from livecellx import sample_data
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.core.single_cell import SingleCellTrajectoryCollection, SingleCellTrajectory
from livecellx.trajectory.feature_extractors import compute_skimage_regionprops, parallelize_compute_features
from livecellx.preprocess.utils import normalize_img_to_uint8
from livecellx.track.btrack_tracker_utils import single_cell_to_btrack_object


class TestBtrack(unittest.TestCase):
    """Test case for btrack integration with sample data."""
    
    def test_track_simple_interface(self):
        """Test the simple interface for btrack tracking."""
        # Load sample data
        dic_dataset, mask_dataset = sample_data.tutorial_three_image_sys()
        
        # Prepare single cell objects
        single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        
        # Compute features for the single cells using the feature_extractors module
        _, single_cells = parallelize_compute_features(
            single_cells,
            compute_skimage_regionprops,
            params={
                "feature_key": "skimage",
                "preprocess_img_func": normalize_img_to_uint8,
                "sc_level_normalize": True,
            },
            replace_feature=True,
            verbose=True
        )
        
        # Create a simple tracking function
        def track(scs):
            """Simple tracking function that takes a list of single cells and returns a trajectory collection."""
            # Convert single cells to btrack objects
            objects = []
            sc_id_counter = 0
            for sc in scs:
                sc.id = sc_id_counter
                sc_id_counter += 1

            for sc in scs:
                # Extract features from the skimage feature dictionary
                feature_names = ['area', 'perimeter', 'eccentricity']
                if 'skimage' in sc.feature_dict:
                    # Create a flattened feature dictionary with skimage features
                    skimage_features = sc.feature_dict['skimage']
                    features = {}
                    for feature_name in feature_names:
                        if feature_name in skimage_features:
                            features[feature_name] = skimage_features[feature_name]
                    
                    # Update the feature_dict with the flattened features
                    sc.feature_dict.update(features)
                
                obj = single_cell_to_btrack_object(sc, feature_names=feature_names)
                objects.append(obj)
            
            # Create and configure the tracker
            config = {
                "motion_model": {
                    "name": "cell_motion",
                    "dt": 1.0,
                    "measurements": 3,  # x, y, z
                    "states": 6,  # x, y, z, dx, dy, dz
                    "accuracy": 7.5,
                    "prob_not_assign": 0.1,
                    "max_lost": 5,
                    "A": {
                        "matrix": [1, 0, 0, 1, 0, 0,
                                  0, 1, 0, 0, 1, 0,
                                  0, 0, 1, 0, 0, 1,
                                  0, 0, 0, 1, 0, 0,
                                  0, 0, 0, 0, 1, 0,
                                  0, 0, 0, 0, 0, 1]
                    },
                    "H": {
                        "matrix": [1, 0, 0, 0, 0, 0,
                                  0, 1, 0, 0, 0, 0,
                                  0, 0, 1, 0, 0, 0]
                    },
                    "P": {
                        "sigma": 150.0,
                        "matrix": [0.1, 0, 0, 0, 0, 0,
                                  0, 0.1, 0, 0, 0, 0,
                                  0, 0, 0.1, 0, 0, 0,
                                  0, 0, 0, 1.0, 0, 0,
                                  0, 0, 0, 0, 1.0, 0,
                                  0, 0, 0, 0, 0, 1.0]
                    },
                    "G": {
                        "sigma": 15.0,
                        "matrix": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
                    },
                    "R": {
                        "sigma": 5.0,
                        "matrix": [1.0, 0, 0,
                                  0, 1.0, 0,
                                  0, 0, 1.0]
                    }
                },
                "optimizer": {
                    "name": "hungarian",
                    "params": {
                        "max_search_radius": 20.0
                    }
                },
                "update_mode": BayesianUpdates.EXACT
            }
            
            # Create the tracker with the configuration
            tracker = btrack.BayesianTracker()
            tracker.configure(config)
            
            # Append the objects to the tracker
            tracker.append(objects)
            
            # Track the objects
            tracker.track()
            
            
            # Get the tracks
            tracks = tracker.tracks
            print("---flag1 # tracks: len(tracks)")
            print(f"Found {len(tracks)} tracks")
            for track in tracks:
                print("length of track: len(track)")
                print(len(track))

            # Create a SingleCellTrajectoryCollection
            traj_collection = SingleCellTrajectoryCollection()
            
            # Create a mapping from (timeframe, cell_id) to original single cell objects
            original_sc_map = {}
            for sc in scs:
                key = (sc.timeframe, sc.id)
                original_sc_map[key] = sc
            
            print("original_sc_map keys:", original_sc_map.keys())
            # Convert tracks to SingleCellTrajectory objects
            for track in tracks:
                track_id = track.ID
                trajectory = SingleCellTrajectory(track_id=track_id, img_dataset=dic_dataset)
                
                # Get the objects in the track
                track_objects = []
                
                # Try different methods to access track objects
                if hasattr(track, 'data') and track.data is not None:
                    # Method 1: Use track.data
                    track_objects = track.data
                elif hasattr(track, 'refs') and track.refs is not None:
                    # Method 2: Use track.refs to get objects from tracker.objects
                    for ref in track.refs:
                        if ref is not None and ref >= 0:
                            try:
                                track_objects.append(tracker.objects[ref])
                            except Exception as e:
                                print(f"Error accessing object with ref {ref}: {e}")
                else:
                    # Method 3: Try to iterate through the track
                    try:
                        track_objects = list(track)
                    except Exception as e:
                        print(f"Error iterating track: {e}")
                
                print(f"{track_id}, number of objects in track: {len(track_objects)}")
                # Add each object in the track to the trajectory
                for obj in track_objects:
                    timeframe = int(obj.t)
                    
                    # Get the original cell ID
                    original_cell_id = obj.ID
                    
                    # Try to find the original single cell object
                    original_sc = None
                    if original_cell_id is not None:
                        key = (timeframe, original_cell_id)
                        if key in original_sc_map:
                            original_sc = original_sc_map[key]
                    
                    assert original_sc is not None, f"Original single cell not found for timeframe={timeframe}, ID={obj.ID}"

                    # Use the original single cell object
                    sc = original_sc
                    
                    # Add the single cell to the trajectory
                    trajectory.timeframe_to_single_cell[timeframe] = sc
                # Add the trajectory to the collection if it has cells
                if len(trajectory.timeframe_to_single_cell) > 0:
                    traj_collection.add_trajectory(trajectory)
            
            return traj_collection
        
        # Track the single cells using the simple interface
        trajectories = track(single_cells)
        
        # Check that the result is a SingleCellTrajectoryCollection
        self.assertIsInstance(trajectories, SingleCellTrajectoryCollection)
        
        # Check that trajectories were created
        self.assertGreater(len(trajectories.track_id_to_trajectory), 0)
        
        # Check that each trajectory has cells
        for track_id, trajectory in trajectories.track_id_to_trajectory.items():
            self.assertGreater(len(trajectory.timeframe_to_single_cell), 0)
        
        # Visualize the trajectories
        plt.figure(figsize=(10, 8))
        
        # Plot the trajectories
        for track_id, trajectory in trajectories.track_id_to_trajectory.items():
            # Get the positions of the cells in the trajectory
            positions = []
            for timeframe in sorted(trajectory.timeframe_to_single_cell.keys()):
                sc = trajectory.timeframe_to_single_cell[timeframe]
                x = (sc.bbox[0] + sc.bbox[2]) / 2
                y = (sc.bbox[1] + sc.bbox[3]) / 2
                positions.append((x, y))
            
            # Plot the trajectory
            if positions:
                x_vals, y_vals = zip(*positions)
                plt.plot(x_vals, y_vals, '-o', label=f'Track {track_id}')
        
        plt.title('Cell Trajectories')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig('btrack_trajectories.png')
        plt.close()
        
        # Check that the figure was created
        self.assertTrue(os.path.exists('btrack_trajectories.png'))


if __name__ == '__main__':
    unittest.main()
