# Comments on code (60/100)

As your project started from existing running implementation, the following
criteria was applied:

- 100 for major modifications + enhanced performance
- 95 for major modifications + different application
- 90 for major modifications (e.g. refactoring, port to different language)
- 80 for minor modifications + custom data I/O
- 70 for minor modifications (e.g. add visualizations, minor architectural changes)
- 50 for running the implementation with minor modifications, reporting
  results.
  
It would've been nice if instead of GRU or LSTM, a vanilla recurrent neural
network was implemented as I suggested before. In this case, it could've been
considered a major modification if combined with other enhancements.

The main two contributions regarding the method are either changing to another
existing pretrained model (which is a very minor change with torch), or running
the code with different options.

For the dataset, the script that was used to modify the dataset is
missing. Also, when searched for "cake" you can see that some words have also
wrongly changed. For example, "piece" turned into "cakece". 

Coding could've followed [PEP8](https://www.python.org/dev/peps/pep-0008/) for
better readability.

# Comments on report (70/100)

The report does not follow the guidelines from the announcement.

## Abstract

Well written abstract. Nicely captures what the project was about, and the
important findings.

## Intro

The claim about GRU and LSTM is wrong. LSTM is older than GRU, and GRU is a
different architecture compared to LSTM. 

## Related Work

This section is not required for the project report.

## Contributions of the original paper

This part is completely missing. 

## Contributions of this project

Sections 3 and 4 corresponds to this.

Figure 1 does not show that LSTMs are better than GRUs. It simply shows the two
architectures. Also, it is hard to say one is better than the other for all
tasks. You are investigating this here.

Some of the figures are actually tables.

## Discussions

Section 4.3 corresponds to this. Figure 7 shows some comparison between the
original model vs the modified one. However, there are multiple things that
changed, e.g. the dataset, and the architecture, making it impossible to
compare against the original architecture. Thus, the comparison amonngst the
new datasets are the only valid ones. 

LSTM+ResNext+Modified data is repeated twice, which I presume is a mistake.

