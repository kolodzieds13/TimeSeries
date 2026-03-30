import os
import numpy as np
import pandas as pd
import Rbeast as rb
from joblib import Parallel, delayed
import warnings

# Suppress Rbeast console spam and runtime warnings for clean parallel execution
warnings.filterwarnings("ignore")

class RbeastForestryAnalyzer:
    """
    A highly robust, Bayesian time series analyzer for forestry biomass.
    Utilizes the Rbeast package to evaluate structural breaks and trend probabilities
    using a 50% Majority Rule framework.
    """
    def __init__(self, 
                 harvest_drop_threshold=0.15, 
                 rapid_increase_thresh=5.0,   
                 steady_increase_thresh=1.5,
                 slight_decrease_thresh=-1.5,
                 steady_decrease_thresh=-5.0):
        
        self.harvest_drop_threshold = float(harvest_drop_threshold)
        self.rapid_increase_thresh = float(rapid_increase_thresh)
        self.steady_increase_thresh = float(steady_increase_thresh)
        self.slight_decrease_thresh = float(slight_decrease_thresh)
        self.steady_decrease_thresh = float(steady_decrease_thresh)

    def _classify_bayesian_trend(self, med_pos, med_neg, med_zero, med_slope):
        """
        Classifies the trend based on the Rbeast posterior probabilities using a 
        50% majority rule, modified by the absolute median slope (Mg/ha/yr).
        Returns the category label and the winning probability.
        """
        # Determine the maximum probability across the three states
        max_prob = max(med_pos, med_neg, med_zero)
        
        # Fallback: if no state achieves 50% confidence, the segment is volatile
        if max_prob < 0.50:
            return "volatile", max_prob

        # Increasing Majority
        if med_pos >= 0.50:
            if med_slope > self.rapid_increase_thresh: return "rapid_increase", max_prob
            elif med_slope >= self.steady_increase_thresh: return "steady_increase", max_prob
            else: return "slight_increase", max_prob
            
        # Decreasing Majority
        elif med_neg >= 0.50:
            if med_slope < self.steady_decrease_thresh: return "rapid_decrease", max_prob
            elif med_slope <= self.slight_decrease_thresh: return "steady_decrease", max_prob
            else: return "slight_decrease", max_prob
            
        # Stable Majority
        else:
            return "stable", max_prob

    def analyze_parcel(self, parcel_id, years, biomass):
        """
        Core analytical pipeline mapping the Rbeast output to the specified CSV template.
        """
        years = np.asarray(years, dtype=float).flatten()
        biomass = np.asarray(biomass, dtype=float).flatten()
        
        mask = ~np.isnan(biomass)
        if np.sum(mask) < 15: 
            return {'uniqID': parcel_id, 'Status': 'Insufficient Data'}
            
        # 1. Reconstruct Temporal Geometry for Rbeast 
        full_years = np.arange(int(np.min(years[mask])), int(np.max(years[mask])) + 1)
        series = pd.Series(index=years[mask], data=biomass[mask])
        series = series.reindex(full_years).interpolate(method='linear')
        y_vals = series.values

        # 2. Execute Bayesian Model Averaging
        try:
            # season='none' explicitly optimized for strictly annual forestry data
            o = rb.beast(y_vals, start=float(full_years[0]), deltat=1.0, 
                         tccp_minmax = [0,10], #default [0,10]
                         tseg_minlength= 3, #default 3 # TRIPPLE CHECK THIS|!!!!!
                         mcmc_seed = 42, 
                         mcmc_burbin= 200, #default 200
                         mcmc_samples=8000, #default 8000
                         season='none', quiet=True)
        except Exception as e:
            return {'uniqID': parcel_id, 'Status': f'Rbeast Failure: {str(e)}'}

        fitted_trend = o.trend.Y
        slopes = o.trend.slp
        pos_probs = o.trend.slpSgnPosPr
        zero_probs = getattr(o.trend, 'slpSgnZeroPr', np.zeros_like(pos_probs))

        # 3. Identify Breakpoints via Bayesian Anticipated Number of Changepoints (ncp)
        ncp_estimate = getattr(o.trend, 'ncp_median', getattr(o.trend, 'ncp', 0))
        ncp = int(np.round(np.nan_to_num(ncp_estimate, nan=0)))
        
        # Extract the overall probability that 'ncp' is the true number of breakpoints
        try:
            ncp_prs_array = np.atleast_1d(o.trend.ncpPr)
            ncp_prob = float(ncp_prs_array[ncp]) if ncp < len(ncp_prs_array) else 0.0
        except Exception:
            ncp_prob = 0.0
            
        bkps_indices = list()
        bkps_prob_mapping = {}
        
        if ncp > 0 and hasattr(o.trend, 'cp'):
            # Extract the years and their specific probabilities directly from cp and cpPr
            cp_years = np.atleast_1d(o.trend.cp)[:ncp]
            cp_probs_arr = np.atleast_1d(getattr(o.trend, 'cpPr', np.zeros_like(cp_years)))[:ncp]
            
            for cp_yr, cp_pr in zip(cp_years, cp_probs_arr):
                if not np.isnan(cp_yr):
                    closest_idx = int(np.argmin(np.abs(full_years - cp_yr)))
                    # Ensure we don't segment at the absolute boundaries
                    if 0 < closest_idx < len(full_years) - 1:
                        # Prevent duplicate indices from overwriting with lower probabilities
                        if closest_idx not in bkps_prob_mapping or cp_pr > bkps_prob_mapping[closest_idx]:
                            bkps_prob_mapping[closest_idx] = float(cp_pr)
                            if closest_idx not in bkps_indices:
                                bkps_indices.append(closest_idx)
                        
            bkps_indices = sorted(bkps_indices)

        segment_slopes = list()
        segment_probs = list()
        segment_trends = list()
        segment_r_squareds = list()
        segment_start_years = list()
        segment_end_years = list()
        
        harvest_years = list()
        trend_change_years = list()
        breakpoint_years = list()
        breakpoint_types = list()
        breakpoint_probs = list()
        
        start_idx = 0
        segments_to_process = bkps_indices + [len(y_vals)]
        
        for end_idx in segments_to_process:
            seg_years = full_years[start_idx:end_idx]
            seg_y = y_vals[start_idx:end_idx]
            seg_fitted = fitted_trend[start_idx:end_idx]
            
            seg_slopes = slopes[start_idx:end_idx]
            seg_pos_probs = pos_probs[start_idx:end_idx]
            seg_zero_probs = zero_probs[start_idx:end_idx]
            n_obs = len(seg_years)
            
            # Disturbance Verification
            if end_idx < len(y_vals):
                drop_ratio = float((fitted_trend[end_idx] - fitted_trend[end_idx-1]) / fitted_trend[end_idx-1])
                bp_year = int(full_years[end_idx])
                breakpoint_years.append(bp_year)
                
                # Retrieve the EXACT probability from the cpPr mapping
                bp_prob = bkps_prob_mapping.get(end_idx, 0.0)
                breakpoint_probs.append(f"{bp_prob:.3f}")
                
                if drop_ratio <= -self.harvest_drop_threshold:
                    harvest_years.append(bp_year)
                    breakpoint_types.append('harvest')
                else:
                    trend_change_years.append(bp_year)
                    # Evaluate the Rbeast majority at the exact breakpoint year
                    bp_pos = float(pos_probs[end_idx])
                    bp_zero = float(zero_probs[end_idx])
                    bp_neg = max(0.0, 1.0 - (bp_pos + bp_zero))
                    
                    if bp_pos >= 0.50:
                        bp_type = 'increase'
                    elif bp_neg >= 0.50:
                        bp_type = 'decrease'
                    elif bp_zero >= 0.50:
                        bp_type = 'stable'
                    else:
                        bp_type = 'volatile'
                        
                    breakpoint_types.append(bp_type)

            # Segment Trend Classification
            if n_obs >= 3:
                med_slope = float(np.median(seg_slopes))
                med_pos = float(np.median(seg_pos_probs))
                med_zero = float(np.median(seg_zero_probs))
                med_neg = max(0.0, 1.0 - (med_pos + med_zero)) # Calculate negative probability
                
                trend_label, winning_prob = self._classify_bayesian_trend(med_pos, med_neg, med_zero, med_slope)
                
                # Pseudo R-squared: Segment Fit vs Variance
                ss_res = np.sum((seg_y - seg_fitted)**2)
                ss_tot = np.sum((seg_y - np.mean(seg_y))**2)
                r_squared = float(1 - (ss_res / ss_tot)) if ss_tot!= 0 else 0.0
            else:
                trend_label = "insufficient_segment_data"
                med_slope = 0.0
                winning_prob = 0.0 
                r_squared = 0.0

            segment_slopes.append(f"{med_slope:.3f}")
            segment_probs.append(f"{winning_prob:.3f}")
            segment_trends.append(trend_label)
            segment_r_squareds.append(f"{r_squared:.3f}")
            segment_start_years.append(str(int(seg_years[0])))
            segment_end_years.append(str(int(seg_years[-1])))
            
            start_idx = end_idx

        # 4. Overall Parcel Synthesis
        year1_value = float(y_vals[0])
        year_final_value = float(y_vals[-1])
        total_change = year_final_value - year1_value
        percent_change = float((total_change / year1_value) * 100) if year1_value!= 0 else 0.0
        cv = float((np.std(y_vals) / np.mean(y_vals)) * 100) if np.mean(y_vals)!= 0 else 0.0
        
        abs_pct = abs(percent_change)
        if abs_pct < 20: trend_magnitude = 'low'
        elif abs_pct <= 50: trend_magnitude = 'moderate'
        else: trend_magnitude = 'extreme'
        
        overall_category = "stable"
        if len(harvest_years) > 0:
            final_segment = segment_trends[-1]
            if final_segment in ['rapid_increase', 'steady_increase', 'slight_increase']:
                overall_category = f"managed_recovery_({len(harvest_years)}_events)"
            else:
                overall_category = "harvested_no_recovery"
        elif len(bkps_indices) == 0:
            overall_category = f"continuous_{segment_trends}"
        else:
            overall_category = "variable_dynamics"
            
        data_quality_flag = 'poor' if len(bkps_indices) > 6 else 'ok'
        
        years_since_last_dist = ""
        is_recovering = False
        if harvest_years:
            years_since_last_dist = int(full_years[-1] - harvest_years[-1])
            if segment_trends[-1] in ['rapid_increase', 'steady_increase', 'slight_increase']:
                is_recovering = True

        return {
            'uniqID': parcel_id,
            'num_breakpoints': len(bkps_indices),
            'num_breakpoints_prob': ncp_prob,
            'num_harvest_events': len(harvest_years),
            'num_trend_changes': len(trend_change_years),
            'breakpoint_years_str': ", ".join(map(str, breakpoint_years)),
            'harvest_years_str': ", ".join(map(str, harvest_years)),
            'trend_change_years_str': ", ".join(map(str, trend_change_years)),
            'breakpoint_types_str': ", ".join(breakpoint_types),
            'breakpoint_prob_str': ", ".join(breakpoint_probs),
            'segment_slopes_str': ", ".join(segment_slopes),
            'segment_prob_str': ", ".join(segment_probs),
            'segment_trends_str': ", ".join(segment_trends),
            'segment_r_squared_str': ", ".join(segment_r_squareds),
            'segment_start_years_str': ", ".join(segment_start_years),
            'segment_end_years_str': ", ".join(segment_end_years),
            'year1_value': year1_value,
            'year_final_value': year_final_value,
            'total_change': total_change,
            'percent_change': percent_change,
            'cv': cv,
            'overall_trend_category': overall_category,
            'trend_magnitude': trend_magnitude,
            'data_quality_flag': data_quality_flag,
            'years_since_last_disturbance': years_since_last_dist,
            'is_recovering': is_recovering,
            'most_recent_segment_slope': segment_slopes[-1] if segment_slopes else "",
            'most_recent_segment_prob': segment_probs[-1] if segment_probs else "",
            # 'most_recent_segment_tau': "N/A (Replaced by Rbeast)", 
            'most_recent_trend_str': segment_trends[-1] if segment_trends else "",
            'most_recent_breakpoint_str': breakpoint_types[-1] if breakpoint_types else ""
        }

def process_forestry_data(input_df, id_col='uniqID', year_col='Year', biomass_col='Biomass', 
                          output_dir='./', output_filename='Rbeast_TimeSeries_Results.csv', 
                          n_jobs=-1, **kwargs):
    """
    User-facing function to orchestrate parallel Rbeast processing and export to CSV.
    """
    analyzer = RbeastForestryAnalyzer(**kwargs)
    
    def process_group(name, group):
        group = group.sort_values(year_col)
        return analyzer.analyze_parcel(
            parcel_id=name, 
            years=group[year_col].values, 
            biomass=group[biomass_col].values
        )

    # Groupby parcel and execute in parallel across all CPU cores
    print(f"Initializing Rbeast processing for {input_df[id_col].nunique()} unique parcels...")
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_group)(name, group) for name, group in input_df.groupby(id_col)
    )
    
    # Format and save output
    final_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    final_df.to_csv(out_path, index=False)
    print(f"Success! Results saved to {out_path}")
    
    return final_df

# ==========================================
# Implementation Example
# ==========================================
if __name__ == "__main__":
    # Example initialization 
    # df_raw = pd.read_csv("path_to_your_raw_dataset.csv")
    
    # Custom thresholds can be passed via **kwargs
    # config = {
    #     'harvest_drop_threshold': 0.15,
    #     'prob_threshold': 0.50,
    #     'rapid_increase_thresh': 5.0
    # }
    
    final_results = process_forestry_data(
        input_df=zonalMod, 
        id_col='uniqID', 
        year_col='year', 
        biomass_col='mean',
        output_dir="D:/TO17/^^ExtraExtra/GEDI/Parcel_Analysis/trendAnalysis/Final/beast",
        output_filename='temp_deleteMe.csv',
        n_jobs=-1,
        # **config
    )
    # pass