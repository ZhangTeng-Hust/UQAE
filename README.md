## Title
Joint prediction and accuracy enhancement （JPAE） methods for deep regression learning considering the relationship between point and intervals

## Description of the files
- 1) _Result_evalute.py_：A function for evaluating the effectiveness of prediction interval estimation and point prediction accuracy
- 2)  _Proposed.py_：The proposed method. Take the Energy task as an example.


## Contribution
- 1) An asynchronous training strategy for joint point-interval prediction model with threshold monitoring and parameter sharing is devised, where interval prediction is distribution-free, and the added parts can be combined with any DRL in plug-in form;
- 2) A fuzzy logic-based point accuracy improvement method is proposed, which can guarantee the calibration of point prediction results with certain interpretability.

## Main idea
A joint DRL prediction and accuracy enhancement method considering the point-interval relationship is proposed, called JPAE, in which the transferability of the point prediction model to the interval estimation model in the shallow parameters and the accuracy enhancement of the point prediction due to the point-interval coupling relationship are considered. The Main idea is as follows.
<div align=center>
<img src=[https://github.com/ZhangTeng-Hust/JPAE/main/IMG/Main idea.png](https://github.com/ZhangTeng-Hust/JPAE/blob/main/IMG/Main%20idea.png)https://github.com/ZhangTeng-Hust/JPAE/blob/main/IMG/Main%20idea.png>
</div>
