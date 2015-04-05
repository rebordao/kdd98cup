# Fund Raising Prediction Model

## Task

“This dataset was used in the 1998 kdd cup data mining competition. It was
collected by PVA, a non-profit organisation which provides programs and 
services for US veterans with spinal cord injuries or disease. They raise
money via direct mailing campaigns. The organisation is interested in lapsed
donors: people who have stopped donating for at least 12 months. The available
dataset contains a record for every donor who received the 1997 mailing and
did not make a donation in the 12 months before that. For each of them it
is given whether and how much they donated as a response to this. Apart from 
that, data are given about the previous and the current mailing campaign,
as well as personal information and the giving history of each lapsed donor. 
Also overlay demographics were added. See the documentation and the data 
dictionary for more information.

Size:

- 191779 records: 95412 training cases and 96367 test cases
- 481 attributes
- 236.2 MB: 117.2 MB training data and 119 MB test data

Carefully read the information available about the dataset. Perform 
exploratory data analysis to get a good feel for the data and prepare
the data for data mining. It will be important to do good feature and
case selection to reduce the data dimensionality. The data mining task
is in the first place to classify people as donors or not. Try at least
2 different classifiers, like for example logistic regression or Naive
Bayes. As an extra, you can go on to predict the amount someone is going
to give. A good way of going about this is described in Zadrozny and
Elkan's paper. The success of a solution can then be assessed by calculating
the profits of a mailing campaign targetting all the test individuals that
are predicted to give more than the cost of sending the mail. The profits
when targetting the entire test set is $10,560,”
Extracted from [here](
http://stanford.edu/~cpiech/cs221/homework/finalProject.html).

## Challenges

This dataset is quite noisy, many attributes have a lot of missing values,
and there are records with formatting errors. An important issue is feature 
selection. There are far too many features, and it will be necessary to 
select the most relevant ones, or to construct our own features.
The training set has around 96,000 examples, but contains only 5% positive
examples. Finally, building a useful model for this dataset is made more
difficult by the fact that there is an inverse relationship between the
probability to donate and the amount donated.


## Strategy

## References

“A short overview of the kdd cup 1998 results is available on-line, 
together with short descriptions of the methods used by the winners,
and those who came third. Learning and Making Decisions When Costs and
Probabilities are Both Unknown (2001) by B. Zadrozny and C. Elkan:
The authors use this dataset as an example of a situation where
misclassification costs depend on the individual. Their approach is
more advanced and better described then the ones of the kdd cup winners.”
Extracted from [here](http://stanford.edu/~cpiech/cs221/homework/finalProject.html).

## Author

[Antonio Rebordao](https://www.linkedin.com/in/rebordao)
