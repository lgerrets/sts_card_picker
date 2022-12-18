## .run processing

- Card names in .run files have no standardized format. I should format those as early as possible in pipeline.
- For now we drop out samples from the dataset that contain unknown cards

## Model training

- I'm padding the entire dataset once to minimize out of distribution but this may be bad for perf and generalize poorly?
- Model does not see upgrades.
- Support configurable stateful experiment parameters
