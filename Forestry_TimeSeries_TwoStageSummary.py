import pandas as pd
import numpy as np
from scipy import stats
import os

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures not available. Install with: pip install ruptures")


def mann_kendall_test(data):
    """
    Perform Mann-Kendall trend test (Optimized via Vectorization).
    """
    n = len(data)
    if n < 3:
        return {'tau': np.nan, 'z_statistic': np.nan, 'p_value': np.nan, 'trend': 'insufficient_data', 's_statistic': np.nan}
    
    # Vectorized sign calculation
    i, j = np.triu_indices(n, k=1)
    s = np.sum(np.sign(data[j] - data[i]))
    
    # Calculate variance handling ties
    unique_data, tp = np.unique(data, return_counts=True)
    g = len(unique_data)
    
    if n == g:  # No ties
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # Ties present
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
    
    if var_s <= 0:
        return {'tau': 0, 'z_statistic': 0, 'p_value': 1.0, 'trend': 'no trend', 's_statistic': s}
    
    # Calculate Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    
    # Determine trend
    if p_value < 0.05:
        if tau > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no trend'
    
    return {
        'tau': tau,
        'z_statistic': z,
        'p_value': p_value,
        'trend': trend,
        's_statistic': s
    }


def detect_two_stage_breakpoints(years, biomass_values, 
                                 harvest_penalty=8,
                                 tuning_factor=1.0,
                                 harvest_min_size=3,
                                 trend_min_size=8,
                                 harvest_drop_threshold=25):
    """
    Two-stage breakpoint detection:
    Stage 1: Detect abrupt changes (harvests) using L1 model
    Stage 2: Detect slope changes using linear model with robust variance estimation
    """
    signal = biomass_values.reshape(-1, 1)
    
    # Stage 1: Harvest detection (conservative - only major drops)
    try:
        algo_harvest = rpt.Pelt(model="l1", min_size=harvest_min_size, jump=1).fit(signal)
        harvest_breaks_idx = algo_harvest.predict(pen=harvest_penalty)
        harvest_breaks_idx = [bp for bp in harvest_breaks_idx if bp < len(years)]
    except Exception as e:
        print(f"Harvest detection error: {e}")
        harvest_breaks_idx = []
    
    # Verify these are actually harvest events
    confirmed_harvest_breaks_idx = []
    confirmed_harvest_drops = []
    
    for bp_idx in harvest_breaks_idx:
        if bp_idx > 0 and bp_idx < len(biomass_values):
            before_biomass = biomass_values[bp_idx - 1]
            after_biomass = biomass_values[bp_idx]
            
            if before_biomass > 0:
                drop_pct = ((before_biomass - after_biomass) / before_biomass) * 100
                if drop_pct > harvest_drop_threshold:
                    confirmed_harvest_breaks_idx.append(bp_idx)
                    confirmed_harvest_drops.append(drop_pct)

    # Stage 2: Trend change detection
    try:
        n_obs = len(biomass_values)
        if n_obs > 1:
            # Use MAD of first-order differences to isolate background noise from major disturbances
            diffs = np.diff(biomass_values)
            mad = np.median(np.abs(diffs - np.median(diffs)))
            if mad == 0:
                mad = 1e-6 
            
            # Robust variance estimation for BIC-style penalty
            sigma_robust = mad / (np.sqrt(2) * 0.6745)
            robust_variance = sigma_robust ** 2
            dynamic_trend_penalty = robust_variance * np.log(n_obs) * tuning_factor
        else:
            dynamic_trend_penalty = 0

        # Build Design Matrix: Response variable (y) first, followed by covariates (x) and an intercept term
        time_index = np.arange(n_obs)
        linear_signal = np.column_stack((biomass_values, time_index, np.ones(n_obs)))

        algo_trend = rpt.Pelt(model="linear", min_size=trend_min_size, jump=1).fit(linear_signal)
        trend_breaks_idx = algo_trend.predict(pen=dynamic_trend_penalty)
        trend_breaks_idx = [bp for bp in trend_breaks_idx if bp < len(years)]
    except Exception as e:
        print(f"Trend detection error: {e}")
        trend_breaks_idx = []
    
    filtered_trend_breaks_idx = []
    
    for trend_bp in trend_breaks_idx:
        is_harvest = False
        if trend_bp > 0 and trend_bp < len(biomass_values):
            before = biomass_values[trend_bp - 1]
            after = biomass_values[trend_bp]
            if before > 0:
                drop = ((before - after) / before) * 100
                if drop > harvest_drop_threshold:
                    is_harvest = True
                    if trend_bp not in confirmed_harvest_breaks_idx:
                        confirmed_harvest_breaks_idx.append(trend_bp)
                        confirmed_harvest_drops.append(drop)
        
        # Proximity filter removed entirely to allow capturing immediate post-disturbance regeneration 
        if not is_harvest:
            filtered_trend_breaks_idx.append(trend_bp)
    
    all_breaks_idx = sorted(set(confirmed_harvest_breaks_idx + filtered_trend_breaks_idx))
    all_breaks_years = [int(years[bp]) for bp in all_breaks_idx]
    harvest_years = [int(years[bp]) for bp in sorted(confirmed_harvest_breaks_idx)]
    trend_change_years = [int(years[bp]) for bp in sorted(filtered_trend_breaks_idx)]
    
    return {
        'all_breakpoints_idx': all_breaks_idx,
        'all_breakpoints_years': all_breaks_years,
        'harvest_breaks_idx': sorted(confirmed_harvest_breaks_idx),
        'harvest_years': harvest_years,
        'harvest_drop_percents': confirmed_harvest_drops,
        'trend_change_breaks_idx': sorted(filtered_trend_breaks_idx),
        'trend_change_years': trend_change_years
    }


def calculate_two_stage_hybrid_trends(years, biomass_values,
                                      harvest_penalty=8,
                                      tuning_factor=1.0,
                                      harvest_min_size=3,
                                      trend_min_size=8,
                                      harvest_drop_threshold=25):
    """
    Calculate trend metrics using two-stage hybrid approach.
    """
    if not RUPTURES_AVAILABLE:
        return {
            'error': 'RUPTURES_not_available',
            'num_breakpoints': np.nan,
            'breakpoint_years_str': '',
            'overall_trend_category': 'error',
            'trend_magnitude': np.nan,
            'data_quality_flag': 'ruptures_unavailable'
        }
    
    mask = ~np.isnan(biomass_values)
    years_clean = np.array(years)[mask]
    biomass_clean = np.array(biomass_values)[mask]
    
    if len(biomass_clean) < 10:
        return {
            'num_breakpoints': np.nan, 'num_harvest_events': 0, 'num_trend_changes': 0,
            'breakpoint_years_str': '', 'harvest_years_str': '', 'trend_change_years_str': '',
            'breakpoint_types_str': '', 'segment_sen_slopes_str': '', 'segment_mk_tau_str': '',
            'segment_mk_pvalue_str': '', 'segment_trends_str': '', 'segment_r_squared_str': '',
            'segment_start_years_str': '', 'segment_end_years_str': '',
            'year1_value': np.nan, 'year_final_value': np.nan, 'total_change': np.nan,
            'percent_change': np.nan, 'cv': np.nan, 'overall_trend_category': 'insufficient_data',
            'trend_magnitude': np.nan, 'data_quality_flag': 'insufficient_data',
            'years_since_last_disturbance': np.nan, 'is_recovering': False,
            'most_recent_segment_slope': np.nan, 'most_recent_segment_tau': np.nan
        }
    
    breakpoint_result = detect_two_stage_breakpoints(
        years_clean, biomass_clean, harvest_penalty, tuning_factor, 
        harvest_min_size, trend_min_size, harvest_drop_threshold
    )
    
    all_breaks_idx = breakpoint_result['all_breakpoints_idx']
    all_breaks_years = breakpoint_result['all_breakpoints_years']
    harvest_years = breakpoint_result['harvest_years']
    trend_change_years = breakpoint_result['trend_change_years']
    
    num_breakpoints = len(all_breaks_years)
    num_harvest_events = len(harvest_years)
    num_trend_changes = len(trend_change_years)
    
    if num_breakpoints > 0:
        segment_boundaries = [0] + all_breaks_idx + [len(years_clean)]
    else:
        segment_boundaries = [0, len(years_clean)]
    
    breakpoint_types = []
    for i, bp_year in enumerate(all_breaks_years):
        if bp_year in harvest_years:
            breakpoint_types.append('harvest')
        else:
            breakpoint_types.append('trend_change')
    
    segment_sen_slopes = []
    segment_mk_tau = []
    segment_mk_pvalue = []
    segment_trends = []
    segment_r_squared = []
    segment_ols_slopes = []
    segment_start_years = []
    segment_end_years = []
    segment_is_post_harvest = []
    
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]
        
        segment_years = years_clean[start_idx:end_idx]
        segment_biomass = biomass_clean[start_idx:end_idx]
        
        if len(segment_years) > 0:
            segment_start_years.append(int(segment_years[0]))
            segment_end_years.append(int(segment_years[-1]))
        else:
            segment_start_years.append(np.nan)
            segment_end_years.append(np.nan)
        
        is_post_harvest = False
        if i > 0:
            prev_break_year = all_breaks_years[i-1]
            if prev_break_year in harvest_years:
                is_post_harvest = True
        segment_is_post_harvest.append(is_post_harvest)
        
        if len(segment_biomass) >= 3:
            mk_result = mann_kendall_test(segment_biomass)
            segment_mk_tau.append(mk_result['tau'])
            segment_mk_pvalue.append(mk_result['p_value'])
            
            # Replace slow Python loop implementation with optimized SciPy theilslopes
            res = stats.theilslopes(segment_biomass, segment_years, alpha=0.95, method='separate')
            segment_sen_slopes.append(res.slope)
            
            ols_slope, intercept, r_value, p_value, std_err = stats.linregress(segment_years, segment_biomass)
            r_squared = r_value ** 2
            segment_r_squared.append(r_squared)
            segment_ols_slopes.append(ols_slope)
            
            if mk_result['p_value'] < 0.05:
                if mk_result['tau'] > 0.3:
                    segment_trends.append('increasing')
                elif mk_result['tau'] < -0.3:
                    segment_trends.append('decreasing')
                elif mk_result['tau'] > 0:
                    segment_trends.append('slight_increase')
                else:
                    segment_trends.append('slight_decrease')
            else:
                segment_trends.append('stable')
        else:
            segment_sen_slopes.append(np.nan)
            segment_mk_tau.append(np.nan)
            segment_mk_pvalue.append(np.nan)
            segment_trends.append('insufficient_data')
            segment_r_squared.append(np.nan)
            segment_ols_slopes.append(np.nan)
    
    year1_value = biomass_clean[0]
    year_final_value = biomass_clean[-1]
    total_change = year_final_value - year1_value
    percent_change = (total_change / year1_value * 100) if year1_value!= 0 else np.nan
    
    mean_biomass = np.mean(biomass_clean)
    std_biomass = np.std(biomass_clean)
    cv = (std_biomass / mean_biomass * 100) if mean_biomass!= 0 else np.nan
    
    years_since_last_disturbance = np.nan
    is_recovering = False
    
    most_recent_segment_slope = segment_sen_slopes[-1] if len(segment_sen_slopes) > 0 else np.nan
    most_recent_segment_tau = segment_mk_tau[-1] if len(segment_mk_tau) > 0 else np.nan
    
    if num_harvest_events > 0:
        last_harvest_year = max(harvest_years)
        years_since_last_disturbance = int(years_clean[-1]) - last_harvest_year
        
        if years_since_last_disturbance >= 3:
            post_harvest_idx = np.where(years_clean > last_harvest_year)[0]
            if len(post_harvest_idx) >= 3:
                post_harvest_biomass = biomass_clean[post_harvest_idx]
                mk_recovery = mann_kendall_test(post_harvest_biomass)
                is_recovering = (mk_recovery['p_value'] < 0.1 and mk_recovery['tau'] > 0.3)
    
    overall_trend_category, trend_magnitude = classify_two_stage_trend(
        segment_trends, segment_sen_slopes, segment_mk_tau,
        segment_is_post_harvest, percent_change, num_harvest_events, 
        num_trend_changes, is_recovering, num_breakpoints
    )
    
    data_quality_flag = assess_two_stage_data_quality(
        num_breakpoints, num_harvest_events, num_trend_changes, 
        segment_r_squared, segment_mk_pvalue, cv, is_recovering
    )
    
    breakpoint_years_str = ', '.join(map(str, all_breaks_years))
    harvest_years_str = ', '.join(map(str, harvest_years))
    trend_change_years_str = ', '.join(map(str, trend_change_years))
    breakpoint_types_str = ', '.join(breakpoint_types)
    segment_sen_slopes_str = ', '.join([f'{s:.3f}' if not np.isnan(s) else 'NA' for s in segment_sen_slopes])
    segment_mk_tau_str = ', '.join([f'{t:.3f}' if not np.isnan(t) else 'NA' for t in segment_mk_tau])
    segment_mk_pvalue_str = ', '.join([f'{p:.4f}' if not np.isnan(p) else 'NA' for p in segment_mk_pvalue])
    segment_trends_str = ', '.join(segment_trends)
    segment_r_squared_str = ', '.join([f'{r:.3f}' if not np.isnan(r) else 'NA' for r in segment_r_squared])
    segment_start_years_str = ', '.join(map(str, segment_start_years))
    segment_end_years_str = ', '.join(map(str, segment_end_years))
    
    return {
        'num_breakpoints': num_breakpoints,
        'num_harvest_events': num_harvest_events,
        'num_trend_changes': num_trend_changes,
        'breakpoint_years_str': breakpoint_years_str,
        'harvest_years_str': harvest_years_str,
        'trend_change_years_str': trend_change_years_str,
        'breakpoint_types_str': breakpoint_types_str,
        'segment_sen_slopes_str': segment_sen_slopes_str,
        'segment_mk_tau_str': segment_mk_tau_str,
        'segment_mk_pvalue_str': segment_mk_pvalue_str,
        'segment_trends_str': segment_trends_str,
        'segment_r_squared_str': segment_r_squared_str,
        'segment_start_years_str': segment_start_years_str,
        'segment_end_years_str': segment_end_years_str,
        'year1_value': year1_value,
        'year_final_value': year_final_value,
        'total_change': total_change,
        'percent_change': percent_change,
        'cv': cv,
        'overall_trend_category': overall_trend_category,
        'trend_magnitude': trend_magnitude,
        'data_quality_flag': data_quality_flag,
        'years_since_last_disturbance': years_since_last_disturbance,
        'is_recovering': is_recovering,
        'most_recent_segment_slope': most_recent_segment_slope,
        'most_recent_segment_tau': most_recent_segment_tau
    }


def classify_two_stage_trend(segment_trends, segment_slopes, segment_tau,
                             segment_is_post_harvest, percent_change,
                             num_harvest_events, num_trend_changes,
                             is_recovering, num_breakpoints):
    abs_percent_change = abs(percent_change) if not np.isnan(percent_change) else 0
    
    if abs_percent_change < 20:
        magnitude = 'low'
    elif abs_percent_change < 50:
        magnitude = 'moderate'
    else:
        magnitude = 'extreme'
    
    if num_breakpoints == 0:
        if len(segment_trends) > 0:
            trend = segment_trends[0]
            if trend == 'increasing':
                return 'steady_growth', magnitude
            elif trend == 'decreasing':
                return 'steady_decline', magnitude
            elif trend in ['slight_increase', 'slight_decrease']:
                return 'weak_trend', magnitude
            else:
                return 'stable', magnitude
        else:
            return 'stable', magnitude
    
    if num_harvest_events > 0:
        if is_recovering:
            if num_harvest_events == 1:
                return 'harvested_recovering', magnitude
            else:
                return f'managed_forest_{num_harvest_events}_cycles', magnitude
        else:
            return 'harvested_no_recovery', magnitude
    
    if num_trend_changes > 0:
        if len(segment_trends) > 0:
            recent_trend = segment_trends[-1]
            if recent_trend == 'increasing':
                return 'variable_increasing', magnitude
            elif recent_trend == 'decreasing':
                return 'variable_decreasing', magnitude
            else:
                return 'highly_variable', magnitude
        else:
            return 'complex_pattern', magnitude
    
    return 'complex_pattern', magnitude


def assess_two_stage_data_quality(num_breakpoints, num_harvest_events,
                                  num_trend_changes, segment_r_squared,
                                  segment_mk_pvalue, cv, is_recovering):
    if num_breakpoints > 6:
        return 'too_many_breakpoints'
    
    if len(segment_r_squared) > 0 and len(segment_mk_pvalue) > 0:
        avg_r_squared = np.nanmean(segment_r_squared)
        significant_segments = sum(1 for p in segment_mk_pvalue if not np.isnan(p) and p < 0.05)
        if avg_r_squared < 0.3 and significant_segments == 0:
            return 'poor_segment_fit'
    
    if cv > 50:
        return 'high_variability'
    
    if num_harvest_events > 0 and not is_recovering:
        return 'no_recovery_after_harvest'
    
    return 'ok'


def create_trend_summary_csv(df, output_path, 
                             parcel_id_col='ParcelID', 
                             year_col='Year', 
                             biomass_col='Biomass_MgHa',
                             harvest_penalty=8,
                             tuning_factor=1.0,
                             harvest_min_size=3,
                             trend_min_size=8,
                             harvest_drop_threshold=25):
    if not RUPTURES_AVAILABLE:
        print("\n" + "="*70)
        print("ERROR: Two-stage approach requires the ruptures Python library")
        print("="*70)
        return None
    
    print(f"Processing {df[parcel_id_col].nunique()} unique parcels using two-stage hybrid approach...")
    trend_results = []
    errors = 0
    
    for i, (parcel_id, group) in enumerate(df.groupby(parcel_id_col)):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} parcels... ({errors} errors)")
        
        group_sorted = group.sort_values(year_col)
        years = group_sorted[year_col].values
        biomass_values = group_sorted[biomass_col].values
        
        metrics = calculate_two_stage_hybrid_trends(
            years, biomass_values, harvest_penalty, tuning_factor,
            harvest_min_size, trend_min_size, harvest_drop_threshold
        )
        
        if 'error' in metrics:
            errors += 1
        
        metrics[parcel_id_col] = parcel_id
        trend_results.append(metrics)
    
    summary_df = pd.DataFrame(trend_results)
    cols = [parcel_id_col] + [col for col in summary_df.columns if col!= parcel_id_col]
    summary_df = summary_df[cols]
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    summary_df.to_csv(output_path, index=False)
    print(f"\nTwo-stage hybrid trend summary saved to: {output_path}")
    return summary_df

###########################################################################################
# Example Usage
###########################################################################################

if __name__ == "__main__":
    if not RUPTURES_AVAILABLE:
        exit(1)
    output_csv_path = 'BDe2_ParcelAnalysis_avgAGBD_MgHa_TimeSeries.csv'
    
    trend_summary = create_trend_summary_csv(
        df=df,
        output_path=output_csv_path,
        parcel_id_col='uniqID', 
        year_col='year', 
        biomass_col='mean',
        harvest_penalty=8,
        tuning_factor=1.0,
        harvest_min_size=5,
        trend_min_size=6,
        harvest_drop_threshold=25
    )
    print("\nProcess completed successfully.")
