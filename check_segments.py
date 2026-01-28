
import h5py
import os

def check_segment_numbering(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Checking segment numbering for: {os.path.basename(file_path)}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            trials = sorted([k for k in f.keys() if k.startswith('trial')])
            
            for trial in trials:
                trial_group = f[trial]
                segments = sorted([k for k in trial_group.keys() if k.startswith('segment')])
                
                print(f"  {trial}: {len(segments)} segments")
                if segments:
                    # Print first and last segment names to see the numbering
                    # Sorting strings 'segment0', 'segment10' can be tricky ('segment10' < 'segment2'),
                    # so we extract numbers.
                    seg_nums = sorted([int(s.replace('segment', '')) for s in segments])
                    print(f"    Range: segment{seg_nums[0]} -> segment{seg_nums[-1]}")
                    print(f"    Sample: {segments[:3]} ... {segments[-3:]}")
                else:
                    print("    (No segments)")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check one of the files mentioned in the mismatch report as an example
    # sub_aaaaaabr.h5 had 76 segments in H5
    target_file = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset/sub_aaaaaabr.h5"
    check_segment_numbering(target_file)
    
    print("-" * 30)
    
    # Check another one
    target_file_2 = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset/sub_aaaaaaax.h5"
    check_segment_numbering(target_file_2)
