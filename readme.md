**Parameter Documentation**

1. --test_n

- Type: int
- Default: None
- Description: Size of the dataset for testing. If not specified, the full dataset will be used.

2. --k

- Type: int

- Default: 256

- Description: Number of samples to select.

3. --prompt

- Type: int

- Default: 8

- Description: Number of prompts.

4. --random (-r)

- Type: Boolean flag (no value required)

- Default: False

- Description: If enabled, randomly selects k samples.

5. --cluster (-c)

- Type: Boolean flag (no value required)

- Default: False

- Description: If enabled, selects k samples using clustering train method.

6. --fast_vote (-f)

- Type: Boolean flag (no value required)

- Default: False

- Description: If enabled, selects k samples using the fast vote method.

7. --vote (-v)

- Type: Boolean flag (no value required)

- Default: False

- Description: If enabled, selects k samples using the vote method.

8. --random_retrieval (-rr)

- Type: Boolean flag (no value required)

- Default: False

- Description: If enabled, retrieves prompts using random retrieval.

9. --similarity_retrieval (-sr)

- Type: Boolean flag (no value required)

- Default: False

- Description: If enabled, retrieves prompts using similarity retrieval.

10. --seed

- Type: int

- Default: 42

- Description: Random seed to ensure reproducibility of experiments.

11. --together (-t)

- Type: Boolean flag (no value required)
- Default: False
- Description: If enabled, uses Together's API.

12. --model 

- Type: string
- Default: "gpt-4o-mini"
- Description: Name of LLM.

**Example: python main.py --test_n=50 --k=256 --prompt=8 -r -c -f -v -rr -sr**