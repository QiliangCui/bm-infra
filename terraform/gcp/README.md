# GCP Project information

### Project information

| folder name                    |      provider project      | provider region    | tpu zone             | tpu name offset | v6e-1 | v6e-4 | v6e-8 | tpu7x-2 | tpu7x-4 | tpu7x-8| tpu7x-16 |
|--------------------------------|----------------------------|--------------------|----------------------|-----------------|-------|-------|-------|---------|---------|---------|---------|
| infer_test_southamerica_west1  | cloud-tpu-inference-test   | southamerica-west1 | southamerica-west1-a | 200             |4      | 0     | 12    |   0     |   0     |   0     |   0     |
| cloud-tpu-inference-test       | cloud-tpu-inference-test   | southamerica-west1 | us-east5-a           | 400             |0      | 0     | 0     |   0     |   0     |   0     |   0     |
| ci_cd                          | cloud-ullm-inference-ci-cd | us-central1        | us-east5-a           | 600             |10     | 2     | 2     |   0     |   0     |   0     |   0     |

`ci_cd` covers every zone in `cloud-ullm-inference-ci-cd`, not one folder per
region. v6e goes in `tpu_zone`, tpu7x in `tpu7x_zone` (us-central1-c), since
`modules/v6e` and `modules/v7x` both take the zone as a variable rather than
reading it from the provider.
