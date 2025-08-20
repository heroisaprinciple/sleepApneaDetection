import boto3
from pathlib import Path

# The bucket is not public.
# Also, IAM and bucket policies are used to upload and delete data only as admins.
# The credentials are properly secured.
# Thus, exposing a bucket name and key patterns has no security risk.

PATIENT_ID = "00001476-100507"
BUCKET = "apnearecordings"
SEGMENT = "seg_009"
KEY = f"{PATIENT_ID}/2025/08/20/13/49/28/recorded_clean.wav"

out_path = Path("fetched_recordings") / PATIENT_ID / SEGMENT / "recorded_segment_clean.wav"
out_path.parent.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

s3.download_file(BUCKET, KEY, str(out_path))
print("Downloaded file")
