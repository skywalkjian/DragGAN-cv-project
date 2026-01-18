import json
import numpy as np

def analyze_full_custom():
    with open('eval_custom_detailed.json', 'r') as f:
        data = json.load(f)
    
    print(f"{'Steps':<10} | {'Method':<30} | {'Mean MD':<15} | {'FID':<15} | {'Valid Cases'}")
    print("-" * 90)
    
    for steps in ["30", "50", "80", "100"]:
        if steps not in data:
            continue
        
        # Get all MDs to identify outliers
        baseline_results = data[steps]['detailed']["Baseline (Nearest Neighbor)"]
        wcat_results = data[steps]['detailed']["WCAT (Context-Aware)"]
        
        # Identify outliers based on lion_g1 (which is > 100)
        # We'll use a threshold or IQR. Let's use IQR for a more scientific approach.
        def get_outliers_indices(values):
             if not values: return set()
             q1 = np.percentile(values, 25)
             q3 = np.percentile(values, 75)
             iqr = q3 - q1
             # Use the standard threshold (1.5 * IQR)
             upper_bound = q3 + 1.5 * iqr
             lower_bound = q1 - 1.5 * iqr
             return {i for i, v in enumerate(values) if v > upper_bound or v < lower_bound}

        baseline_mds = [r['md'] for r in baseline_results]
        wcat_mds = [r['md'] for r in wcat_results]
        
        baseline_outliers = get_outliers_indices(baseline_mds)
        wcat_outliers = get_outliers_indices(wcat_mds)
        
        # Union of outliers to keep comparison fair
        all_outliers = baseline_outliers.union(wcat_outliers)
        
        for method in ["Baseline (Nearest Neighbor)", "WCAT (Context-Aware)"]:
            results = data[steps]['detailed'][method]
            # Filter out outliers
            filtered_mds = [r['md'] for i, r in enumerate(results) if i not in all_outliers]
            
            avg_md = np.mean(filtered_mds)
            fid = data[steps]['fid'][method]
            print(f"{steps:<10} | {method:<30} | {avg_md:<15.4f} | {fid:<15.4f} | {len(filtered_mds)}/16")
        
        # Print which cases were removed
        if all_outliers:
            removed_names = [baseline_results[i]['name'] for i in all_outliers]
            print(f"Removed outliers: {', '.join(removed_names)}")
        
        # Detailed table for each case
        print("\nDetailed breakdown per case:")
        print(f"{'Case Name':<20} | {'Baseline MD':<15} | {'WCAT MD':<15} | {'Status'}")
        print("-" * 65)
        for i in range(len(baseline_results)):
            name = baseline_results[i]['name']
            b_md = baseline_results[i]['md']
            w_md = wcat_results[i]['md']
            status = "Outlier" if i in all_outliers else "Normal"
            print(f"{name:<20} | {b_md:<15.4f} | {w_md:<15.4f} | {status}")
        print("-" * 90)

if __name__ == "__main__":
    analyze_full_custom()
