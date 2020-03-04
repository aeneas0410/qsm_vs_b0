## Update on QSM labels (ground truth) vs B0 segmentation

### - Quantitative analysis

#### Case 1. B0 segmentation using the pre-trained model with 29 B0 manual labels (None: failure)
|    patient_id   | CMD_L   | CMD_R   | MSD_L   | MSD_R   | DC_L   | DC_R   | VOL_L   | VOL_R   |   VOL_manual_L |   VOL_manual_R |
|:-------------|:--------|:--------|:--------|:--------|:-------|:-------|:--------|:--------|---------------:|---------------:|
|  FA005        | 3.518   | 3.833   | 2.925   | 3.265   | 0.215  | 0.3    | 330.7   | 253.9   |         1098.4 |         1086.6 |
|  FA010        | None    | None    | None    | None    | None   | None   | None    | None    |         1045.3 |          915.4 |
|  FA013        | None    | 5.065   | None    | 5.519   | None   | 0.114  | None    | 35.4    |          631.9 |          584.7 |
|  FA014        | 2.131   | 2.194   | 1.813   | 1.45    | 0.602  | 0.643  | 496.1   | 620.1   |          956.7 |          885.8 |
|  FA015        | 0.804   | 2.112   | 1.167   | 1.836   | 0.674  | 0.673  | 372.1   | 513.8   |          679.1 |          732.3 |
|  FA016        | 6.081   | 5.394   | 4.445   | 3.8     | 0.346  | 0.383  | 135.8   | 183.1   |          614.2 |          525.6 |
|  FA017        | 3.305   | 4.274   | 2.504   | 3.192   | 0.366  | 0.346  | 307.1   | 330.7   |          596.5 |          726.4 |
|  FA021        | 1.997   | 22.282  | 2.005   | 17.174  | 0.652  | 0.0    | 265.8   | 643.7   |          513.8 |          614.2 |
|  FA022        | None    | None    | None    | None    | None   | None   | None    | None    |          549.2 |          484.3 |
|  FA024_1      | 2.178   | 2.173   | 1.602   | 1.726   | 0.68   | 0.685  | 442.9   | 543.3   |          738.2 |          750   |
|  FA024_2      | 2.153   | 2.618   | 1.509   | 1.957   | 0.621  | 0.403  | 566.9   | 655.5   |          649.6 |          720.5 |

#### Case 2. B0 segmentation using a trained model with QSM manual labels and B0 images (leave-one-out on 11 new cases); Similar to 7T-3T segmentation (7T guided 3T segmentation), I call this QSM guided B0 segmentation.

|       |    CMD_L |     CMD_R |     MSD_L |     MSD_R |      DC_L |       DC_R |    VOL_L |    VOL_R |   VOL_manual_L |   VOL_manual_R |
|:------|---------:|----------:|----------:|----------:|----------:|-----------:|---------:|---------:|---------------:|---------------:|
| count | 11       | 11        | 11        | 11        | 11        | 11         |   11     |   11     |         11     |         11     |
| **mean**  |  **2.161** |  **1.487**  |  **1.461**  |  **1.104**  |  **0.603** |  **0.675**     |  **918.58** |  **857.92** |        **733.9**   |        **729.62** |
| std   |  1.668 |  0.788 |  0.841 |  0.338 |  0.150 |  0.079 |  237.03 |  177.75 |        203.90 |        179.72 |
| min   |  0.453   |  0.378    |  0.853    |  0.735    |  0.297    |  0.502     |  466.5   |  637.8   |        513.8   |        484.3   |
| 25%   |  1.1845  |  1.028    |  1.003    |  0.9215   |  0.5305   |  0.636     |  776.55  |  776.6   |        605.35  |        599.45  |
| 50%   |  1.619   |  1.557    |  1.168    |  1.071    |  0.635    |  0.687     |  956.7   |  850.4   |        649.6   |        726.4   |
| 75%   |  2.62    |  1.6475   |  1.4645   |  1.1095   |  0.701    |  0.7275    | 1071.85  |  894.7   |        847.45  |        817.9   |
| 90%   |  3.332   |  1.831    |  1.785    |  1.372    |  0.772    |  0.754     | 1133.9   | 1003.9   |       1045.3   |        915.4   |
| max   |  6.394   |  3.456    |  3.846    |  1.974    |  0.788    |  0.774     | 1322.9   | 1275.6   |       1098.4   |       1086.6   |


#### Case 3. QSM segmentation using a trained model with QSM manual labels and QSM images (leave-one-out on 11 new cases); This proves the importance of better representation on dentate nuclei for deep learning.
|       |     CMD_L |     CMD_R |     MSD_L |     MSD_R |       DC_L |       DC_R |    VOL_L |    VOL_R |   VOL_manual_L |   VOL_manual_R |
|:------|----------:|----------:|----------:|----------:|-----------:|-----------:|---------:|---------:|---------------:|---------------:|
| count | 11        | 11        | 11        | 11        | 11         | 11         |   11     |   11     |         11     |         11     |
| **mean**  |  **0.755** |  **0.786** |  **0.565** |  **0.476** |  **0.861**     |  **0.862**  |  **781.69** |  **789.74** |        **733.9**   |        **729.62** |
| std   |  0.483657 |  0.331469 |  0.240696 |  0.180795 |  0.0196672 |  0.0391071 |  243.605 |  214.295 |        203.904 |        179.718 |
| min   |  0.294    |  0.163    |  0.359    |  0.288    |  0.832     |  0.805     |  490.2   |  478.4   |        513.8   |        484.3   |
| 25%   |  0.556    |  0.6685   |  0.429    |  0.38     |  0.844     |  0.8385    |  617.15  |  682.1   |        605.35  |        599.45  |
| 50%   |  0.652    |  0.853    |  0.482    |  0.424    |  0.864     |  0.856     |  767.7   |  750     |        649.6   |        726.4   |
| 75%   |  0.731    |  0.946    |  0.623    |  0.514    |  0.871     |  0.8815    |  800.2   |  871.05  |        847.45  |        817.9   |
| 90%   |  0.912    |  0.988    |  0.716    |  0.643    |  0.882     |  0.921     | 1009.9   | 1104.3   |       1045.3   |        915.4   |
| max   |  2.134    |  1.394    |  1.197    |  0.929    |  0.895     |  0.924     | 1381.9   | 1181.1   |       1098.4   |       1086.6   |


### - Failure analysis for cases 1 and 2

A model pre-trained with existing 29 datasets from our database (B0 and B0 labels) (case 1) produced much worse segmentation (and even failures) than a model trained with new data (B0 and QSM labels). This might be attributed to the different intensity distribution between training data (our database) and test data (unseen cases). 

Note that training a model with QSM labels to segment B0 images is a similar scenario to 3T segmentation using 7T labels we have done for STN segmentation.

When training a model with QSM labels, using B0 image for segmentation (case 2) was much worse than using QSM image (case 3). This is quite reasonable but current QSM guided B0 segmentation is not good enough to justify why we used only B0 image in this work. This can be due to the discrepancy between QSM manual labels and B0 hyper-intense appearance (i.e., manual labels on the B0) that might be affected by incorrect registration between QSM and B0, uncertainty of manual labeling on the QSM, or a sub-optimal trained model. Indeed, I found such issue in many cases below. Minimizing such discrepancy is important because a deep learning model relies on the B0 image appearance within the QSM label. 

#### 1. Major issues in case 1: different intensity distribution between training data (from the existing 29 data and new data)

##### : image intensity histograms for one of 29 training data and one out of new datasets below are different, especially in lower and higher intensity values
     
   ![Histogram of FA022 B0 image out of new datasets](/FA022_hist.jpg)
   ![Histogram of SLEEP126 B0 image out of the existing datasets](/SLEEP126_hist.jpg)
   
#### 2. Major issues in case 2

##### : Discrepancy between QSM manual labels and B0 hyper-intense appearance -> this might be due to registration error or image contrast: FA005 (see the difference in left dentate below), FA014, FA015, FA017, FA021, FA022, FA024_1, FA024_2 (blue: QSM labels, red: B0 segmentation, green: QSM segmentation)

<img src="/FA005_qsm_labels_and_b0_seg_on_b0.jpg" width="400" height="249" /> <img src="/FA005_qsm_labels_and_b0_seg_on_qsm.jpg" width="400" height="249" />
   
##### : Uncertainty of manually labeling on the QSM - FA013 (see some artifacts on the left dentate of the left image)
<img src="/FA013_qsm_label_left.jpg" width="400" height="249" /> 

##### : Insufficient segmentation due to training on mismatched QSM labels and b0 appearance (3 might be affected by 1 and 2)- FA010 (see under-segmentation on the B0 image (left) and over-segmentation on the QSM image (right) below), FA013, FA015, FA016, FA022 (blue: QSM labels, red: B0 segmentation, green: QSM segmentation)

<img src="/FA010_qsm_labels_and_b0_seg_on_b0.jpg" width="400" height="249" /> <img src="/FA010_qsm_labels_and_b0_seg_on_qsm.jpg" width="400" height="249" />

To further improve QSM guided B0 segmentation, we might need to further verify registration between QSM and B0 to better represent B0 features, especially around the DCN. 

The reviewers might point out that it would be better to use QSM images for more accurate segmentation although it requires a post-processing. Indeed, many other segmentation works use QSM images these days (A quick idea for the future work is to generate QSM like image from B0 image using a deep learning model to improve segmentation).
