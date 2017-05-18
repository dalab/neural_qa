# neural_qa
Source code for "Neural Multi-Step Reasoning for Question Answering on Semi-Structured Tables"

The code base is slightly convoluted, if you have any questions do not hesitate to contact: till_haug@hotmail.com

# Generate Training and Test Data
- Get code from Compositional Semantic Parsing on Semi-Structured Tables (ACL 2015) (https://worksheets.codalab.org/worksheets/0xf26cd79d4d734287868923ad1067cf4c/)
- Overwrite code from 1) with code from src_neural
- Run code according to instructions from 1), during run files starting with "translated_" will be created, they will contain the training and test data.
  Note: Scala is a requirement

# Create Vocabulary
- Run ./vocab_creator.py to create the vocabulary based on the training and test data (needs to be given as one file as input).

# Training / Testing
See main.py.
Note: for testing make sure to train a model first and specifiy the persisted model in the parameters.