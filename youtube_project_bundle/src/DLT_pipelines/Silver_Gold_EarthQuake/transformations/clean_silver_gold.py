from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# =============================================================================
# Requirement 1: Monthly Earthquake Trend (Fact Table)
# =============================================================================

@dp.materialized_view(
    name="youtube_dev.gold.fact_earthquake_month_count",
    comment="Monthly earthquake trend aggregations with magnitude and tsunami statistics"
)
def fact_earthquake_month_count():
    """
    Aggregate earthquake data by year and month.
    Calculates total events, max/avg magnitude, avg depth, and tsunami warnings.
    """
    df = spark.read.table("youtube_dev.silver.earthquake_data_final")
    
    result = df.groupBy("event_year", "event_month").agg(
        F.count("*").alias("total_earthquakes"),
        F.max("magnitude").alias("max_magnitude"),
        F.round(F.avg("magnitude"), 2).alias("avg_magnitude"),
        F.round(F.avg("depth_km"), 2).alias("avg_depth_km"),
        F.sum(F.when(F.col("tsunami") == 1, 1).otherwise(0)).alias("total_tsunami_warnings")
    ).orderBy(F.col("event_year").desc(), F.col("event_month").desc())
    
    return result


# =============================================================================
# Requirement 2: Daily High-Frequency Monitoring (Fact Table)
# =============================================================================

@dp.materialized_view(
    name="youtube_dev.gold.fact_earthquake_day_count",
    comment="Daily earthquake monitoring with high-frequency event tracking"
)
def fact_earthquake_day_count():
    """
    Aggregate earthquake data by date.
    Tracks daily event counts, max magnitude, and tsunami warnings.
    """
    df = spark.read.table("youtube_dev.silver.earthquake_data_final")
    
    result = df.groupBy("event_year", "event_month", "event_day", "event_date").agg(
        F.count("*").alias("daily_total_events"),
        F.max("magnitude").alias("daily_max_magnitude"),
        F.sum(F.when(F.col("tsunami") == 1, 1).otherwise(0)).alias("daily_tsunami_warnings")
    ).orderBy(F.col("event_date").desc())
    
    return result


# =============================================================================
# Requirement 3: Magnitude Destructiveness Classification (Dimension Table)
# =============================================================================

@dp.materialized_view(
    name="youtube_dev.gold.dim_earthquake_magnitude_class_stats",
    comment="Earthquake classification by magnitude destructiveness levels"
)
def dim_earthquake_magnitude_class_stats():
    """
    Classify earthquakes by magnitude and aggregate by year.
    Categories: Micro, Light, Strong, Major.
    """
    df = spark.read.table("youtube_dev.silver.earthquake_data_final")
    
    # Add magnitude classification
    df_classified = df.withColumn(
        "magnitude_class",
        F.when(F.col("magnitude") < 3.0, "Micro (微震)")
        .when((F.col("magnitude") >= 3.0) & (F.col("magnitude") < 5.0), "Light (轻震)")
        .when((F.col("magnitude") >= 5.0) & (F.col("magnitude") < 7.0), "Strong (强震)")
        .when(F.col("magnitude") >= 7.0, "Major (大地震)")
        .otherwise("Unknown")
    )
    
    result = df_classified.groupBy("event_year", "magnitude_class").agg(
        F.count("*").alias("event_count")
    ).orderBy(F.col("event_year").desc(), F.col("magnitude_class").asc())
    
    return result


# =============================================================================
# Requirement 4: Disaster-Level High-Risk Alerts (Fact Detail Table)
# =============================================================================

@dp.materialized_view(
    name="youtube_dev.gold.fact_earthquake_high_risk_alerts",
    comment="High-risk earthquake alerts filtered by magnitude, alert level, or tsunami"
)
def fact_earthquake_high_risk_alerts():
    """
    Filter high-risk earthquakes based on:
    - Magnitude >= 6.0
    - Alert level is orange or red
    - Tsunami warning (tsunami == 1)
    """
    df = spark.read.table("youtube_dev.silver.earthquake_data_final")
    
    # Filter for high-risk conditions
    high_risk = df.filter(
        (F.col("magnitude") >= 6.0) |
        (F.col("alert").isin(["orange", "red"])) |
        (F.col("tsunami") == 1)
    )
    
    # Select relevant columns
    result = high_risk.select(
        "id",
        "event_time",
        "place",
        "magnitude",
        "depth_km",
        "alert",
        "tsunami",
        "url"
    ).orderBy(F.col("event_time").desc())
    
    return result


# =============================================================================
# Requirement 5: Semantic BI View for Monthly Report
# =============================================================================

@dp.materialized_view(
    name="youtube_dev.gold.view_semantic_bi_monthly_report",
    comment="BI-friendly semantic view with renamed columns for monthly earthquake reporting"
)
def view_semantic_bi_monthly_report():
    """
    Create a semantic view for BI tools based on monthly fact table.
    Transforms column names to business-friendly format.
    """
    df = spark.read.table("youtube_dev.gold.fact_earthquake_month_count")
    
    result = df.select(
        # Concatenate year and month into YYYY-MM format
        F.concat_ws(
            "-",
            F.col("event_year").cast("string"),
            F.lpad(F.col("event_month").cast("string"), 2, "0")
        ).alias("Reporting_Month"),
        F.col("total_earthquakes").alias("Total_Seismic_Events"),
        F.col("max_magnitude").alias("Highest_Magnitude_Recorded"),
        F.col("avg_magnitude").alias("Average_Magnitude"),
        F.col("total_tsunami_warnings").alias("Tsunami_Alert_Count")
    ).orderBy(F.col("Reporting_Month").desc())
    
    return result