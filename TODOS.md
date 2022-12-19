## .run processing

- Card names in .run files have no standardized format. I should format those as early as possible in pipeline.
- For now we drop out samples from the dataset that contain unknown cards
- I'm looking into vanilla (not Run History Plus) .run files. There is even more work to do with them, but I have some idea of how to do it, with some forward/backward logic.

## Model training

- I'm padding the entire dataset once to minimize out of distribution but this may be bad for perf and generalize poorly?
- Support configurable stateful experiment parameters
- Validation dataset
