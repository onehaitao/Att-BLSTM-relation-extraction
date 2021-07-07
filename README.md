# Att-BLSTM-relation-extraction
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-based-bidirectional-long-short-term/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=attention-based-bidirectional-long-short-term)

Implementation of [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://www.aclweb.org/anthology/P16-2034.pdf).

## Environment Requirements
* python 3.6
* pytorch 1.3.0

## Data
* [SemEval2010 Task8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50) \[[paper](https://www.aclweb.org/anthology/S10-1006.pdf)\]
* [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/)

## Usage
1. Download the embedding and decompress it into the `embedding` folder.
2. Run the following the commands to start the program.
```shell
python run.py
```
More details can be seen by `python run.py -h`.

3. You can use the official scorer to check the final predicted result.
```shell
perl semeval2010_task8_scorer-v1.2.pl proposed_answer.txt predicted_result.txt >> result.txt
```

## Result
The result of my version and that in paper are present as follows:
| paper | my version |
| :------: | :------: |
| 0.840 | 0.8313 |

The training log can be seen in `train.log` and the official evaluation results is available in `result.txt`.

*Note*:
* Some settings may be different from those mentioned in the paper.
* No validation set used during training.


## Reference Link