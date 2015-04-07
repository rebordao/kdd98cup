# KDD Cup 98 Challenge

## Task

For a [direct mailing campaign organised by a non-profit organisation](
http://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98doc.txt),
build statistical models that:

1. Identify the recipients that will engage with the campaign.
2. Maximise the campaign’s revenue.

My solutions to these tasks are in scripts `donors.py` and `profits.py`,
respectively. The technical report is at `report.html`. All the code and the
report are available in a [github repository](
https://github.com/rebordao/kdd98cup).

## Dataset

This dataset was used in the [KDD Cup 98 Challenge](
http://www.sigkdd.org/kdd-cup-1998-direct-marketing-profit-optimization). It
was collected by a non-profit organisation that helps US Veterans. They
raise money via direct mailing campaigns.

See the [documentation and the data dictionary](
https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) for more information.

The profits when targeting the entire testset are $10,560. The cost of
sending each mail is $0.68.

#### Size

- 191779 records: 95412 training cases and 96367 test cases
- 481 attributes
- 236.2 MB: 117.2 MB training data and 119 MB test data

## My Solutions

They are structured around the following steps:

1. Data Importation
2. Exploratory Analysis
3. Data Munging
4. Feature Selection
5. Model Selection
6. Training
7. Testing
8. Model Evaluation and Comparison

In my solution to task 1 I follow this procedure.

In my solution to task 2 first I predict who is a donor, and then - using just
those samples - I train a classifier that predicts how much the person donated.
Then I mail all the ones where the prediction is higher than $0.68.

I used only the training cases that were provided and made my training and test
sets out of that file. Thus my train and test sets together have 95412 cases.

## System Architecture

```
.
├── README.md
├── config.yml
├── data
│   ├── cup98LRN.csv
│   └── cup98lrn.zip
├── donors.py
├── lib
│   ├── __init__.py
│   ├── analyser.py
│   ├── importer.py
│   ├── preprocessor.py
│   ├── utils.py
├── profits.py
├── report.html
└── report.md
```

The main files are `donors.py` and `profits.py`. The project’s configuration
is at `config.yml` and all the auxiliary classes and their methods are in `lib`.

## Author

[Antonio Rebordao](https://www.linkedin.com/in/rebordao) 2015
