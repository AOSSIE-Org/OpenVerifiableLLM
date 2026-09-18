# Separate ordered tokenization benchmark

Original full preparation was allowed to finish unchanged. In separate fresh processes,
the first4096 complete eligible articles (23528808 targets) were passed through the
unchanged prepare_stream serializer and text_ids roundtrip/uint16 checks. Serial
encoding took22.180989271s. Ordered encode_batch, at most128documents/8MiB
per batch (oversized single article retained whole), eight Rayon threads, took
9.872589923s. All four output files, including stream manifest, have exactly
equal full-file hashes. See serial.json and batch128.json.

This is a2.25x local bounded-prefix observation, not a complete-corpus throughput forecast
or reconstruction/G02 acceptance. Production preparation code and environment were
not changed. Future adoption needs a separately bound recipe and complete reconstruction.

Initial orchestration failed before output creation because the parent directory was
absent; after creating it, the first serial stream completed but report serialization
rejected floating-point timing fields. Outputs were preserved; fresh-v2 measurements
use decimal strings and reran the full selected prefix. No failed timing was credited.
