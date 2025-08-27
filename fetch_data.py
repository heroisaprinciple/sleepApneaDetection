import boto3
from pathlib import Path

# The bucket is not public.
# Also, IAM and bucket policies are used to upload and delete data only as admins.
# The credentials are properly secured.
# Thus, exposing a bucket name and key patterns has no security risk.

PATIENT_ID = "00001486-100507"
BUCKET = "apnearecordings"
SEGMENT = "seg_000"
KEY = f"{PATIENT_ID}/2025/08/21/15/20/04/recorded_raw.wav"
EVENT = "non_apnea" # could be apnea

out_path = Path("fetched_recordings") / PATIENT_ID / EVENT / SEGMENT / "recorded_segment_raw.wav"
out_path.parent.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

s3.download_file(BUCKET, KEY, str(out_path))
print("Downloaded file")
