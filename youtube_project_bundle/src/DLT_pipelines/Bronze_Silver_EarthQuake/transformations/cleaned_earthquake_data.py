from pyspark import pipelines as dp
from pyspark.sql.functions import col, explode, from_unixtime, current_timestamp, expr, year, month, dayofmonth, to_date, date_format
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, LongType

# Define explicit schema for GeoJSON structure
geojson_schema = StructType([
    StructField("type", StringType(), True),
    StructField("features", ArrayType(StructType([
        StructField("type", StringType(), True),
        StructField("properties", StructType([
            StructField("mag", DoubleType(), True),
            StructField("place", StringType(), True),
            StructField("time", LongType(), True),
            StructField("updated", LongType(), True),
            StructField("tz", LongType(), True),
            StructField("url", StringType(), True),
            StructField("detail", StringType(), True),
            StructField("felt", DoubleType(), True),
            StructField("cdi", DoubleType(), True),
            StructField("mmi", DoubleType(), True),
            StructField("alert", StringType(), True),
            StructField("status", StringType(), True),
            StructField("tsunami", LongType(), True),
            StructField("sig", DoubleType(), True),
            StructField("net", StringType(), True),
            StructField("code", StringType(), True),
            StructField("ids", StringType(), True),
            StructField("sources", StringType(), True),
            StructField("types", StringType(), True),
            StructField("nst", DoubleType(), True),
            StructField("dmin", DoubleType(), True),
            StructField("rms", DoubleType(), True),
            StructField("gap", DoubleType(), True),
            StructField("magType", StringType(), True),
            StructField("type", StringType(), True),
            StructField("title", StringType(), True)
        ]), True),
        StructField("geometry", StructType([
            StructField("type", StringType(), True),
            StructField("coordinates", ArrayType(DoubleType()), True)
        ]), True),
        StructField("id", StringType(), True)
    ])), True)
])

# Step 1: Read raw JSON data from Bronze layer using Auto Loader
@dp.temporary_view()
def bronze_earthquake_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("multiLine", "true")
        .schema(geojson_schema)
        .load("/Volumes/youtube_dev/bronze/earthquake_data")
    )

# Step 2: Transform and flatten the data
@dp.temporary_view()
def earthquake_flattened():
    df = spark.readStream.table("bronze_earthquake_raw")
    
    # Explode features array to get individual earthquake records
    df_exploded = df.select(explode(col("features")).alias("feature"))
    
    # Extract and flatten all fields
    df_transformed = df_exploded.select(
        # Extract id
        col("feature.id").alias("id"),
        
        # Extract properties
        col("feature.properties.mag").cast(DoubleType()).alias("magnitude"),
        col("feature.properties.place").alias("place"),
        
        # --- 完美时间解析模块 ---
        col("feature.properties.time").alias("time_epoch"), # 原始发生时间戳
        col("feature.properties.updated").alias("updated_epoch"), # 原始更新时间戳 (给你补回来了！)
        from_unixtime(col("feature.properties.time") / 1000).cast("timestamp").alias("event_time"),
        from_unixtime(col("feature.properties.updated") / 1000).cast("timestamp").alias("updated_time"),
        to_date(from_unixtime(col("feature.properties.time") / 1000)).alias("event_date"),
        year(from_unixtime(col("feature.properties.time") / 1000)).alias("event_year"),
        month(from_unixtime(col("feature.properties.time") / 1000)).alias("event_month"),
        dayofmonth(from_unixtime(col("feature.properties.time") / 1000)).alias("event_day"),
        # -------------------------
        
        col("feature.properties.tz").alias("timezone"),
        col("feature.properties.url").alias("url"),
        col("feature.properties.detail").alias("detail"),
        col("feature.properties.felt").cast(DoubleType()).alias("felt"),
        col("feature.properties.cdi").cast(DoubleType()).alias("cdi"),
        col("feature.properties.mmi").cast(DoubleType()).alias("mmi"),
        col("feature.properties.alert").alias("alert"),
        col("feature.properties.status").alias("status"),
        col("feature.properties.tsunami").alias("tsunami"),
        col("feature.properties.sig").cast(DoubleType()).alias("significance"),
        col("feature.properties.net").alias("network"),
        col("feature.properties.code").alias("code"),
        col("feature.properties.ids").alias("ids"),
        col("feature.properties.sources").alias("sources"),
        col("feature.properties.types").alias("types"),
        col("feature.properties.nst").cast(DoubleType()).alias("num_stations"),
        col("feature.properties.dmin").cast(DoubleType()).alias("dmin"),
        col("feature.properties.rms").cast(DoubleType()).alias("rms"),
        col("feature.properties.gap").cast(DoubleType()).alias("gap"),
        col("feature.properties.magType").alias("magnitude_type"),
        col("feature.properties.type").alias("event_type"),
        col("feature.properties.title").alias("title"),
        
        # Extract geometry
        col("feature.geometry.coordinates").getItem(0).cast(DoubleType()).alias("longitude"),
        col("feature.geometry.coordinates").getItem(1).cast(DoubleType()).alias("latitude"),
        col("feature.geometry.coordinates").getItem(2).cast(DoubleType()).alias("depth_km"),
        
        # Add load timestamp for auditing
        current_timestamp().alias("_load_timestamp")
    )
    
    return df_transformed

# Step 3: Create target streaming table
dp.create_streaming_table(
    name="earthquake_data_final",
    comment="Clean earthquake data from Bronze layer with flattened GeoJSON structure"
)

# Step 4: Apply CDC to maintain the target table with SCD Type 1
dp.create_auto_cdc_flow(
    target="earthquake_data_final",
    source="earthquake_flattened",
    keys=["id"],
    sequence_by="_load_timestamp",
    stored_as_scd_type=1
)
