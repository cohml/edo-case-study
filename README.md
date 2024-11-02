# Overview

This repository contains the work done for a take-home assignment for a
prospective employer.

The assignment can be broken down into two parts: a modeling task, and a data
storage system design task. This repository contains a dedicated README file
and subdirectory for each part.

# Assignment Instructions

The original instructions for this assignment are as follows:

> Before your interview, please prepare a Jupyter notebook (and optionally
  modules you import into the notebook) to load up synthetic data from this
  link, examine a handful of descriptive statistics, do a bit of modeling
  (specific questions are listed in the interview outline below), and design a
  storage class. Please send the notebook (and any custom modules) no later
  than 24 hours before your follow-up interview is scheduled. This can be sent
  to <email-redacted> and <email-redacted>.
>
> This interview consists of two parts: a modeling exercise and a design
  exercise. The expectation is that this will take a few hours, and much of the
  interview would be talking about what you’d do next, so it’s not expected
  that you’ll cover every possible angle of each question. In that vein, feel
  free to implement the simplest/easiest solution, and during the interview you
  can describe what fancier or more complex things you might do in a real
  project. I’m most interested in seeing how you think about answering the
  questions and what approaches you take (or would take with more time).
>
> During the interview, you'll walk us through your code, and we’ll talk
  through how you approach the problem and what next steps you’d explore if you
  had more time.
>
> Here’s the outline of that interview — there are a lot of parts to squeeze
  into an hour, so we’ve tagged how much time we’ll want to spend on each
  chunk:
>
> **Modeling Exercise**
>
> *Data files in data.zip:*
>   - `people.csv`
>     - one row per person
>     - columns: person_id and a handful of demographics
>   - `exposures.csv`
>     - one row per ad exposure event
>     - columns: `person`, `time`, `ad_id`, `channel_id`
>       - `person` contains unique IDs for individual people.
>       - `time` contains timestamps at which people were exposed to ads.
>       - Different `ad_id`’s represent different versions of the ad.
>       - Different `channel_id`s represent different TV networks where this ad
          aired.
>   - `site_visits.csv`
>     - one row per website visit event
>     - columns: `person`, `time`
>       - `person` contains unique IDs for individual people.
>       - `time` contains timestamps at which people visited the website.
>
> *Goal:*
>   - Understand patterns of exposure (by person, demographics, ad, and
      channel)
>
> *Visualize the data (10 mins):*
>   - Person-level
>     - How often are people exposed?
>     - How often do people visit the website?
>     - How are the demographics distributed?
>
> *Models of exposure (10 mins):*
> - Use the demographic data to model the total number of exposures for each
    person.
> - What is the distribution of predicted exposure values?
>
> *Example discussion topics (15 mins):*
>   - How would you test out of sample fit, robustness?
>   - What should we track about this model if we wanted to deploy it to
      production?
>
> **Design Exercise (15 min)**
>
> **Goal:**
>   - Design a storage service that will live within an existing modeling
      pipeline. The storage service should be able to read and write tabular
      data as well as arbitrary python objects. The storage service should
      support various backends such as NFS, S3, or database.
>   - An example use case of the storage service:
>     - Save a `pandas.DataFrame` to a database
>     - Load a `pandas.DataFrame` from a database
>   - Design a generic storage service (interface, protocol, ABC, etc.)
>   - Implement a storage service object that uses a local filesystem.
>   - Implement unit tests for your local storage service.
