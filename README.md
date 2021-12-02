# PreProcMachine

Machine-learning powered dataset preprocessing - we will try to automate the data preprocessing phase of the machine learning workflow.

### Abstract

-   (An undertaking of this size is potentially outside the scope of this project; in the case that it is, we will try to proceed with one of the other propose projects). Preprocessing and data cleaning are the activities that consume the largest amounts of time in a machine learning workflow. Further, when humans perform this task they generally follow a set of rules and reactions to determine what operations to perform on a given dataset to clean and improve it. Given that this seems like something that could be modeled by something like a decision tree, we would like to attempt to apply machine learning to this problem in order to automate it.

### Approach

-   Create a dataset of datasets and the training steps used to preprocess them - run these datasets through a metadata generator to fill out attributes for our model - try to figure out which attributes impact preprocessing procedure the most as well as how best to represent the “optimal” preprocessing routing in our dataset (ordinal series data?) - run the dataset through the model.

### Persona

-   Use case for this utility would be for machine learning users looking to expedite their preprocessing routine for a new dataset while retaining some configurability in the process. Options will be given (tentatively) for things like “maintaining data readability” or “prioritize size reduction” and the like.

### Dataset Links

-   Given the relatively novel and meta nature of this problem, we would potentially have to construct our own dataset. This would involve either downloading datasets from kaggle in bulk, or finding a way to scrape datasets from sources on the internet. These datasets would each then be run through a labeller program to derive dataset characteristics and metadata, which would then be put into a master dataset.
