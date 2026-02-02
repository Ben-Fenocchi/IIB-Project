# run_ilostat_pull.py

from ilostat_client import ILOSTATClient, ILOSTATConfig

# Initialise client
ilo = ILOSTATClient(
    ILOSTATConfig(
        raw_dir="External databases/ILOSTAT/data/raw",
        derived_dir="External databases/ILOSTAT/data/derived",
    )
)

# Download + read indicator table
df = ilo.read_table(
    "EMP_DWAP_NOC_RT_A",
    directory="indicator"
)

print(df.shape)
print(df.columns[:15])

# Save outputs (parquet + csv)
ilo.save_outputs(
    df,
    "ilostat_EMP_DWAP_NOC_RT_A_raw"
)
