import json
import numpy as np

def analyze():
    with open('/root/DragGAN-cv-project/eval_results_detailed.json', 'r') as f:
        data = json.load(f)

    steps = ["30", "50", "80"]
    methods = ["Baseline (Nearest Neighbor)", "WCAT (Context-Aware)"]

    print(f"{'Steps':<10} | {'Method':<30} | {'Mean MD':<15} | {'FID':<15} | {'Valid Cases'}")
    print("-" * 90)

    for s in steps:
        if s not in data:
            continue
        
        # Get all MDs to identify outliers
        baseline_results = data[s]['detailed']["Baseline (Nearest Neighbor)"]
        wcat_results = data[s]['detailed']["WCAT (Context-Aware)"]
        
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
        
        # Union of outliers
        all_outliers = baseline_outliers.union(wcat_outliers)
        
        for m in methods:
            results = data[s]['detailed'][m]
            filtered_mds = [r['md'] for i, r in enumerate(results) if i not in all_outliers]
            
            avg_md = np.mean(filtered_mds)
            fid = data[s]['fid'][m]
            print(f"{s:<10} | {m:<30} | {avg_md:<15.4f} | {fid:<15.4f} | {len(filtered_mds)}/{len(results)}")
        
        if all_outliers:
            removed_names = [baseline_results[i]['name'] for i in all_outliers]
            print(f"Removed outliers: {', '.join(removed_names)}")
        print("-" * 90)

if __name__ == "__main__":
    analyze()
