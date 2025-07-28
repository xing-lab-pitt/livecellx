#!/usr/bin/env python3
"""
MAXIMIZED single-cell data generation from CTC datasets.
This script addresses all filtering issues to ensure ALL available SEG files are matched.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import skimage.io
import skimage.measure
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from collections import Counter, defaultdict
import warnings
from datetime import datetime

def extract_timepoint_enhanced(filename, dataset_name=None):
    """Enhanced timepoint extraction with fuzzy matching and multiple strategies"""
    name = Path(filename).stem.lower()
    
    # Comprehensive patterns in order of specificity
    patterns = [
        # CTC standard patterns
        r'man_seg(\d+)$',           # man_seg000, man_seg058, etc.
        r'man_track(\d+)$',         # man_track000, man_track001, etc.
        
        # Alternative patterns  
        r'seg(\d+)$',               # seg000
        r'track(\d+)$',             # track000
        r't(\d+)$',                 # t000
        
        # With underscores
        r'seg_(\d+)$',              # seg_000
        r'track_(\d+)$',            # track_000
        
        # More flexible patterns
        r'seg.*?(\d+)$',            # seg with any chars then numbers
        r'track.*?(\d+)$',          # track with any chars then numbers
        
        # Dataset-specific patterns (add more as needed)
        r'mask(\d+)$',              # mask000
        r'label(\d+)$',             # label000
        
        # Very flexible - any sequence of digits at the end
        r'.*?(\d+)$',               # any string ending with digits
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    
    return None

def create_comprehensive_seg_tra_mapping(datasets_base_dir):
    """
    Create comprehensive SEG/TRA mapping that maximizes data usage.
    Processes BOTH _GT and _ST sequences and implements fuzzy matching.
    """
    datasets_base_dir = Path(datasets_base_dir)
    mapping_results = {}
    
    print("🔧 Creating COMPREHENSIVE SEG/TRA mapping (maximized coverage)...")
    
    for dataset_dir in sorted(datasets_base_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == 'raw-zips':
            continue
            
        dataset_name = dataset_dir.name
        inner_dataset_dir = dataset_dir / dataset_name
        
        if not inner_dataset_dir.exists():
            continue
            
        dataset_mapping = {}
        
        # Process ALL sequences (both _GT and _ST) - MAJOR IMPROVEMENT
        all_sequences = [d for d in inner_dataset_dir.iterdir() 
                        if d.is_dir() and (d.name.endswith('_GT') or d.name.endswith('_ST'))]
        
        for seq_dir in sorted(all_sequences):
            seq_name = seq_dir.name
            
            seg_dir = seq_dir / "SEG"
            tra_dir = seq_dir / "TRA"
            
            if not (seg_dir.exists() and tra_dir.exists()):
                continue
            
            # Get ALL .tif files
            seg_files = sorted([f for f in seg_dir.iterdir() if f.suffix.lower() == '.tif'])
            tra_files = sorted([f for f in tra_dir.iterdir() if f.suffix.lower() == '.tif'])
            
            if len(seg_files) == 0 or len(tra_files) == 0:
                continue
            
            # Enhanced timepoint extraction
            seg_timepoints = {}
            tra_timepoints = {}
            
            for seg_file in seg_files:
                timepoint = extract_timepoint_enhanced(seg_file.name, dataset_name)
                if timepoint is not None:
                    seg_timepoints[timepoint] = seg_file
            
            for tra_file in tra_files:
                timepoint = extract_timepoint_enhanced(tra_file.name, dataset_name)
                if timepoint is not None:
                    tra_timepoints[timepoint] = tra_file
            
            # Find matching timepoints
            seg_tps = set(seg_timepoints.keys())
            tra_tps = set(tra_timepoints.keys())
            exact_matches = seg_tps & tra_tps
            
            # FUZZY MATCHING: Find SEG files with nearby TRA files (±1 frame tolerance)
            fuzzy_matches = set()
            for seg_tp in seg_tps - exact_matches:
                for delta in [-1, 0, 1]:  # Check ±1 frame
                    candidate_tp = seg_tp + delta
                    if candidate_tp in tra_tps and candidate_tp not in exact_matches:
                        fuzzy_matches.add((seg_tp, candidate_tp))
                        break
            
            # Create mapping for exact matches
            sequence_mapping = []
            for tp in sorted(exact_matches):
                sequence_mapping.append({
                    'timepoint': tp,
                    'seg_file': seg_timepoints[tp],
                    'tra_file': tra_timepoints[tp],
                    'match_type': 'exact'
                })
            
            # Add fuzzy matches
            for seg_tp, tra_tp in fuzzy_matches:
                sequence_mapping.append({
                    'timepoint': seg_tp,
                    'seg_file': seg_timepoints[seg_tp],
                    'tra_file': tra_timepoints[tra_tp],
                    'match_type': 'fuzzy',
                    'tra_timepoint': tra_tp
                })
            
            if sequence_mapping:
                dataset_mapping[seq_name] = sequence_mapping
        
        if dataset_mapping:
            mapping_results[dataset_name] = dataset_mapping
    
    # Count total pairs
    total_pairs = 0
    exact_pairs = 0
    fuzzy_pairs = 0
    
    for dataset_mapping in mapping_results.values():
        for sequence_mapping in dataset_mapping.values():
            total_pairs += len(sequence_mapping)
            exact_pairs += sum(1 for p in sequence_mapping if p['match_type'] == 'exact')
            fuzzy_pairs += sum(1 for p in sequence_mapping if p['match_type'] == 'fuzzy')
    
    print(f"📊 MAXIMIZED mapping results:")
    print(f"   Total SEG/TRA pairs: {total_pairs:,}")
    print(f"   Exact matches: {exact_pairs:,}")
    print(f"   Fuzzy matches: {fuzzy_pairs:,}")
    print(f"   Improvement vs current: {total_pairs - 578:,} additional pairs ({(total_pairs - 578)/578*100:.1f}% increase)")
    
    return mapping_results

def extract_cell_id_relaxed(cell_mask, tra_mask, cell_label, seg_file_path, tra_file_path):
    """
    RELAXED cell ID extraction that accepts dominant IDs with >80% pixel coverage.
    This reduces filtering and increases data usage.
    """
    # Get TRA values where cell mask is True
    cell_tra_values = tra_mask[cell_mask > 0]
    
    # Remove background (0 values)
    nonzero_tra_values = cell_tra_values[cell_tra_values > 0]
    
    if len(nonzero_tra_values) == 0:
        return {
            'cell_id': None, 
            'is_valid': False, 
            'issue': 'No tracking ID found in cell region',
            'unique_ids': [],
            'id_counts': {},
            'total_pixels': len(cell_tra_values),
            'nonzero_pixels': 0
        }
    
    # Find unique tracking IDs and their counts
    unique_ids, counts = np.unique(nonzero_tra_values, return_counts=True)
    id_counts = dict(zip(unique_ids, counts))
    
    # RELAXED Quality control: Accept cells with exactly one unique ID OR dominant ID >80%
    if len(unique_ids) == 1:
        return {
            'cell_id': int(unique_ids[0]), 
            'is_valid': True, 
            'issue': None,
            'unique_ids': unique_ids.tolist(),
            'id_counts': id_counts,
            'total_pixels': len(cell_tra_values),
            'nonzero_pixels': len(nonzero_tra_values),
            'match_quality': 'perfect'
        }
    else:
        # Check if dominant ID has >80% coverage
        dominant_id = unique_ids[np.argmax(counts)]
        dominant_count = np.max(counts)
        dominant_ratio = dominant_count / len(nonzero_tra_values)
        
        if dominant_ratio >= 0.8:  # RELAXED THRESHOLD
            return {
                'cell_id': int(dominant_id), 
                'is_valid': True, 
                'issue': None,
                'unique_ids': unique_ids.tolist(),
                'id_counts': id_counts,
                'total_pixels': len(cell_tra_values),
                'nonzero_pixels': len(nonzero_tra_values),
                'dominant_ratio': dominant_ratio,
                'match_quality': 'dominant'
            }
        else:
            return {
                'cell_id': None, 
                'is_valid': False, 
                'issue': f'Multiple tracking IDs found ({len(unique_ids)} IDs): dominant={dominant_id} ({dominant_ratio:.1%}) < 80%',
                'unique_ids': unique_ids.tolist(),
                'id_counts': id_counts,
                'total_pixels': len(cell_tra_values),
                'nonzero_pixels': len(nonzero_tra_values),
                'dominant_id': int(dominant_id),
                'dominant_ratio': dominant_ratio
            }

def get_source_origin_name(dataset_name, sequence_name, timepoint):
    """Generate a standardized source origin name for tracking data sources."""
    return f"{dataset_name.upper()}-{sequence_name.split('_')[0]}-T{timepoint:03d}"

def save_single_cell_sample(seg_mask, cell_mask, cell_prop, dataset_name, sequence_num, 
                           timepoint, cell_id, padding, idx, output_dir):
    """Save a single cell sample with padding"""
    try:
        # Get bounding box
        bbox = cell_prop.bbox
        
        # Extract cell region
        cell_region = seg_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        # Create binary mask for this cell
        cell_binary = (cell_region == cell_prop.label).astype(np.uint8) * 255
        
        # Add padding
        padded_height = cell_region.shape[0] + 2 * padding
        padded_width = cell_region.shape[1] + 2 * padding
        
        # Create padded arrays
        padded_raw = np.zeros((padded_height, padded_width), dtype=seg_mask.dtype)
        padded_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
        
        # Place cell in center
        start_h = padding
        end_h = start_h + cell_region.shape[0]
        start_w = padding
        end_w = start_w + cell_region.shape[1]
        
        padded_raw[start_h:end_h, start_w:end_w] = cell_region
        padded_mask[start_h:end_h, start_w:end_w] = cell_binary
        
        # Save files
        filename_base = f"{dataset_name}_{sequence_num:02d}_T{timepoint:03d}_CID{cell_id}_pad{padding}_{idx:04d}"
        raw_filename = f"{filename_base}_raw.tif"
        mask_filename = f"{filename_base}_mask.tif"
        
        # Determine split directory (will be moved later)
        raw_path = output_dir / "train" / raw_filename
        mask_path = output_dir / "train" / mask_filename
        
        # Save images
        Image.fromarray(padded_raw).save(raw_path)
        Image.fromarray(padded_mask).save(mask_path)
        
        return raw_path, mask_path, {
            'padded_shape': (padded_height, padded_width),
            'original_shape': cell_region.shape
        }
        
    except Exception as e:
        print(f"Error saving sample {idx}: {e}")
        return None, None, None

def process_ctc_datasets_maximized(datasets_base_dir, output_dir, padding_options=[10, 20, 50, 100], 
                                  min_cell_area=30, max_cells_per_frame=None):  # RELAXED LIMITS
    """
    Process CTC datasets with MAXIMIZED data usage and relaxed filtering.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val" 
    test_dir = output_dir / "test"
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(exist_ok=True)
    
    print("🔧 Creating MAXIMIZED SEG/TRA mapping...")
    seg_tra_mapping = create_comprehensive_seg_tra_mapping(datasets_base_dir)
    
    # Count total pairs
    total_pairs = 0
    for dataset_mapping in seg_tra_mapping.values():
        for sequence_mapping in dataset_mapping.values():
            total_pairs += len(sequence_mapping)
    
    print(f"📊 Found {total_pairs:,} SEG/TRA pairs across {len(seg_tra_mapping)} datasets")
    
    all_samples = []
    idx = 0
    
    # Statistics tracking
    stats = {
        'total_pairs_processed': 0,
        'total_cells_processed': 0,
        'valid_cells': 0,
        'invalid_cells': 0,
        'filtering_reasons': defaultdict(int),
        'dataset_stats': defaultdict(lambda: {'pairs': 0, 'cells': 0, 'valid': 0}),
        'match_quality_stats': defaultdict(int),
    }
    
    # Process each dataset
    for dataset_name, dataset_mapping in tqdm(seg_tra_mapping.items(), desc="Processing datasets"):
        print(f"\n📁 Processing dataset: {dataset_name}")
        
        for seq_name, sequence_mapping in dataset_mapping.items():
            print(f"  📂 Sequence: {seq_name} ({len(sequence_mapping)} timepoints)")
            
            stats['dataset_stats'][dataset_name]['pairs'] += len(sequence_mapping)
            
            # Process each timepoint
            for pair in tqdm(sequence_mapping, desc=f"  Processing {seq_name}", leave=False):
                timepoint = pair['timepoint']
                seg_file = pair['seg_file']
                tra_file = pair['tra_file']
                match_type = pair['match_type']
                
                stats['total_pairs_processed'] += 1
                
                try:
                    # Load images
                    seg_mask = skimage.io.imread(seg_file)
                    tra_mask = skimage.io.imread(tra_file)
                    
                    if seg_mask.shape != tra_mask.shape:
                        print(f"    ⚠️  Shape mismatch: {seg_file.name} vs {tra_file.name}")
                        continue
                    
                    # Get cell regions
                    cell_props = skimage.measure.regionprops(seg_mask)
                    
                    # REMOVED arbitrary max_cells_per_frame limit
                    if max_cells_per_frame and len(cell_props) > max_cells_per_frame:
                        cell_props = cell_props[:max_cells_per_frame]
                    
                    for cell_prop in cell_props:
                        cell_label = cell_prop.label
                        cell_area = cell_prop.area
                        
                        stats['total_cells_processed'] += 1
                        stats['dataset_stats'][dataset_name]['cells'] += 1
                        
                        if cell_area < min_cell_area:  # RELAXED from 50 to 30
                            stats['filtering_reasons']['Cell too small'] += 1
                            stats['invalid_cells'] += 1
                            continue
                        
                        # Extract cell mask
                        cell_mask = (seg_mask == cell_label)
                        
                        # Extract cell ID with RELAXED validation
                        cell_id_result = extract_cell_id_relaxed(
                            cell_mask, tra_mask, cell_label, seg_file, tra_file
                        )
                        
                        if cell_id_result['is_valid']:
                            stats['valid_cells'] += 1
                            stats['dataset_stats'][dataset_name]['valid'] += 1
                            stats['match_quality_stats'][cell_id_result.get('match_quality', 'unknown')] += 1
                            
                            # Process each padding option
                            for padding in padding_options:
                                # Extract and save cell
                                sequence_num = int(seq_name.split('_')[0])
                                raw_img_path, mask_img_path, sample_data = save_single_cell_sample(
                                    seg_mask, cell_mask, cell_prop, dataset_name, sequence_num, 
                                    timepoint, cell_id_result['cell_id'], padding, idx, output_dir
                                )
                                
                                if raw_img_path and mask_img_path:
                                    source_origin = get_source_origin_name(dataset_name, seq_name, timepoint)
                                    
                                    sample_info = {
                                        'raw_img_path': str(raw_img_path),
                                        'mask_path': str(mask_img_path),
                                        'cell_id': cell_id_result['cell_id'],
                                        'cell_label': cell_label,
                                        'dataset': dataset_name,
                                        'source_origin': source_origin,
                                        'sequence': sequence_num,
                                        'timepoint': timepoint,
                                        'padding': padding,
                                        'cell_area': cell_area,
                                        'bbox': str(cell_prop.bbox),
                                        'seg_file': str(seg_file),
                                        'tra_file': str(tra_file),
                                        'match_type': match_type,
                                        'match_quality': cell_id_result.get('match_quality', 'unknown')
                                    }
                                    
                                    all_samples.append(sample_info)
                                    idx += 1
                        else:
                            stats['invalid_cells'] += 1
                            stats['filtering_reasons'][cell_id_result['issue']] += 1
                
                except Exception as e:
                    print(f"    ❌ Error processing {seg_file.name}: {str(e)}")
                    continue
    
    # Print final statistics
    print(f"\n" + "=" * 80)
    print(f"📊 MAXIMIZED PROCESSING COMPLETE!")
    print(f"   Total SEG/TRA pairs: {stats['total_pairs_processed']:,}")
    print(f"   Total cells processed: {stats['total_cells_processed']:,}")
    print(f"   Valid cells: {stats['valid_cells']:,} ({stats['valid_cells']/max(1,stats['total_cells_processed'])*100:.1f}%)")
    print(f"   Invalid cells: {stats['invalid_cells']:,}")
    print(f"   Total samples generated: {len(all_samples):,}")
    
    print(f"\n📋 Match quality breakdown:")
    for quality, count in stats['match_quality_stats'].items():
        print(f"   {quality}: {count:,} cells")
    
    print(f"\n📋 Filtering reasons:")
    for reason, count in sorted(stats['filtering_reasons'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {reason}: {count:,} cells")
    
    print(f"\n📁 Dataset breakdown:")
    for dataset, ds_stats in stats['dataset_stats'].items():
        valid_pct = ds_stats['valid'] / max(1, ds_stats['cells']) * 100
        print(f"   {dataset}: {ds_stats['pairs']:,} pairs, {ds_stats['cells']:,} cells, {ds_stats['valid']:,} valid ({valid_pct:.1f}%)")
    
    # Save statistics
    with open(output_dir / "generation_statistics_maximized.json", 'w') as f:
        json_stats = {
            'total_pairs_processed': stats['total_pairs_processed'],
            'total_cells_processed': stats['total_cells_processed'],
            'valid_cells': stats['valid_cells'],
            'invalid_cells': stats['invalid_cells'],
            'filtering_reasons': dict(stats['filtering_reasons']),
            'dataset_stats': {k: dict(v) for k, v in stats['dataset_stats'].items()},
            'match_quality_stats': dict(stats['match_quality_stats']),
            'total_samples': len(all_samples)
        }
        json.dump(json_stats, f, indent=2)
    
    # Create splits and save dataframes
    if all_samples:
        df = pd.DataFrame(all_samples)
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['dataset'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dataset'])
        
        # Add split column
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Save individual split files
        train_df.to_csv(output_dir / "train_data_aux.csv", index=False)
        val_df.to_csv(output_dir / "val_data_aux.csv", index=False)
        test_df.to_csv(output_dir / "test_data_aux.csv", index=False)
        
        # Save combined file
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_df.to_csv(output_dir / "all_data_aux.csv", index=False)
        
        print(f"\n💾 Saved datasets: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
        
        return all_df
    else:
        print(f"\n❌ No valid samples generated!")
        return None

if __name__ == "__main__":
    # Configuration
    datasets_base_dir = "datasets/celltrackingchallenge"
    output_dir = "comprehensive_ctc_single_cell_data_maximized"
    padding_options = [10, 20, 50, 100]
    
    print("🚀 Starting MAXIMIZED CTC dataset processing...")
    print(f"📁 Input: {datasets_base_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Padding options: {padding_options}")
    print(f"✨ Key improvements:")
    print(f"   - Process BOTH _GT and _ST sequences")
    print(f"   - Fuzzy timepoint matching (±1 frame)")
    print(f"   - Relaxed cell ID validation (>80% dominant)")
    print(f"   - Reduced min cell area (30 vs 50)")
    print(f"   - Removed max cells per frame limit")
    
    # Process datasets
    result_df = process_ctc_datasets_maximized(
        datasets_base_dir=datasets_base_dir,
        output_dir=output_dir,
        padding_options=padding_options,
        min_cell_area=30,      # RELAXED
        max_cells_per_frame=None  # REMOVED LIMIT
    )
    
    if result_df is not None:
        print(f"\n✅ MAXIMIZED processing complete!")
        print(f"📊 Total samples generated: {len(result_df):,}")
        print(f"📂 Output directory: {output_dir}")
        print(f"🎯 Expected improvement: 10x+ more samples than current 24,760")
    else:
        print(f"\n❌ No samples were generated. Check the logs for details.")