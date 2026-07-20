from analysis import process_analysis

analysis_df = process_analysis(
    vacuum,
    screw_rpm,
    melt_pressure,
    moisture,
    zone2_temp,
    zone3_temp,
    puller_speed
)