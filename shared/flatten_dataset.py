"""Flatten dataset structure: move sequences from class subdirs to root."""
import os
import shutil

data_root = r"c:\Projects\My AI Brain\AI 2nd Brain\03 Projects\AIC-Tracker\OSTrack_Model\OSTrack\data\comp\dataset3"

print(f"Flattening dataset at: {data_root}")

# Get all items in dataset3 root
for item in os.listdir(data_root):
    item_path = os.path.join(data_root, item)
    
    if not os.path.isdir(item_path):
        continue
    
    # Check if this is a flat sequence (has img/ and groundtruth.txt directly)
    has_img = os.path.exists(os.path.join(item_path, 'img'))
    has_gt = os.path.exists(os.path.join(item_path, 'groundtruth.txt'))
    
    if has_img and has_gt:
        # Already flat sequence, skip
        continue
    
    # This is a class folder with sequences inside - flatten it
    print(f"\nProcessing class folder: {item}")
    
    sequences_to_move = []
    for subitem in os.listdir(item_path):
        subitem_path = os.path.join(item_path, subitem)
        if os.path.isdir(subitem_path):
            has_sub_img = os.path.exists(os.path.join(subitem_path, 'img'))
            has_sub_gt = os.path.exists(os.path.join(subitem_path, 'groundtruth.txt'))
            if has_sub_img and has_sub_gt:
                sequences_to_move.append(subitem)
    
    # Move sequences to root
    for seq_name in sequences_to_move:
        src = os.path.join(item_path, seq_name)
        dst = os.path.join(data_root, seq_name)
        
        # Check if destination already exists
        if os.path.exists(dst):
            print(f"  [skip] {seq_name} already at root")
            continue
        
        try:
            shutil.move(src, dst)
            print(f"  [moved] {seq_name}")
        except Exception as e:
            print(f"  [error] {seq_name}: {e}")
    
    # Remove empty class folder
    try:
        if not os.listdir(item_path):
            os.rmdir(item_path)
            print(f"  [removed] empty folder: {item}")
    except Exception as e:
        print(f"  [error removing folder] {item}: {e}")

print("\nFlattening complete!")
