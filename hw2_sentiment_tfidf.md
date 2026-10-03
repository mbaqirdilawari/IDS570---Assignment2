# Homework 2: Sentiment Analysis and TF-IDF
Muhammad Baqir

## Setup

Loading packages.

``` r
library(readr)     # reading text files
library(dplyr)     # editing tables
library(tidyr)     # reshaping tables
library(stringr)   # find-and-replace on text
library(tidytext)  # tokenizing, stopwords, Bing lexicon
library(quanteda)  # DFM and TF-IDF
```

## Part I: Raw-Count Sentiment Analysis

### Step 1: Tokenize and clean the text

Reading all three texts. The third text (A06785) is needed later for
TF-IDF, so all three are cleaned together using the same decisions.

``` r
circle_raw <- read_file("texts/A07594__Circle_of_Commerce.txt")
free_raw   <- read_file("texts/B14801__Free_Trade.txt")
third_raw  <- read_file("texts/A06785.txt")

texts_all <- tibble(
  doc_title = c("Circle of Commerce", "Free Trade", "A06785"),
  text      = c(circle_raw, free_raw, third_raw)
)
```

Normalizing the text: long s to s, collapsing extra whitespace,
lowercasing.

``` r
texts_all <- texts_all %>%
  mutate(
    text_norm = text %>%
      str_replace_all("ſ", "s") %>%
      str_replace_all("\\s+", " ") %>%
      str_to_lower()
  )

# Preview: the first 300 characters of each text, before and after cleaning
texts_all %>%
  mutate(
    before = str_sub(text, 1, 300),
    after  = str_sub(text_norm, 1, 300)
  ) %>%
  select(doc_title, before, after) %>%
  print(width = Inf)
```

    # A tibble: 3 × 3
      doc_title         
      <chr>             
    1 Circle of Commerce
    2 Free Trade        
    3 A06785            
      before                                                                        
      <chr>                                                                         
    1 THE CIRCLE OF COMMERCE. The Prooeme. HERODOTVS in his CLIO, reportes that CRO…
    2 CAP. I. The Causes of the want of Money in England. IT hauing pleased God to …
    3 STrong imagination, nourished by opinion, may cause most men to suppose, that…
      after                                                                         
      <chr>                                                                         
    1 the circle of commerce. the prooeme. herodotvs in his clio, reportes that cro…
    2 cap. i. the causes of the want of money in england. it hauing pleased god to …
    3 strong imagination, nourished by opinion, may cause most men to suppose, that…

Tokenizing into one row per word, removing stopwords, and keeping only
words made of plain letters (this removes numbers and stray symbols).

``` r
data("stop_words")

tokens_clean <- texts_all %>%
  select(doc_title, text_norm) %>%
  unnest_tokens(word, text_norm) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_detect(word, "^[a-z]+$"))

# Preview:
head(tokens_clean, 10)
```

    # A tibble: 10 × 2
       doc_title          word     
       <chr>              <chr>    
     1 Circle of Commerce circle   
     2 Circle of Commerce commerce 
     3 Circle of Commerce prooeme  
     4 Circle of Commerce herodotvs
     5 Circle of Commerce clio     
     6 Circle of Commerce reportes 
     7 Circle of Commerce croesvs  
     8 Circle of Commerce king     
     9 Circle of Commerce lydia    
    10 Circle of Commerce sonne    

Checking the result: the number of words per document, and confirming
that no long s remains.

``` r
tokens_clean %>% count(doc_title)
```

    # A tibble: 3 × 2
      doc_title              n
      <chr>              <int>
    1 A06785             12733
    2 Circle of Commerce 12972
    3 Free Trade          8024

``` r
sum(str_detect(tokens_clean$word, "ſ"))
```

    [1] 0

**Preprocessing decisions (used for both Part I and Part II):**

- The long s (ſ) was converted to s.
- Extra whitespace was collapsed and all text was lowercased.
- Standard tidytext stopwords were removed.
- Only tokens, made entirely of letters were kept. This removes numbers
  and stray symbols.

After cleaning, Circle of Commerce has 12,972 words, Free Trade has
8,024, and A06785 has 12,733.

### Step 2: Join tokens to the Bing sentiment dictionary

Labeling each word as positive or negative using the Bing lexicon. Only
words found in Bing are kept.

``` r
bing <- get_sentiments("bing")

tokens_sentiment <- tokens_clean %>%
  inner_join(bing, by = "word", relationship = "many-to-many")
  
# Preview:
head(tokens_sentiment, 10)
```

    # A tibble: 10 × 3
       doc_title          word        sentiment
       <chr>              <chr>       <chr>    
     1 Circle of Commerce danger      negative 
     2 Circle of Commerce death       negative 
     3 Circle of Commerce ready       positive 
     4 Circle of Commerce violent     negative 
     5 Circle of Commerce danger      negative 
     6 Circle of Commerce kill        negative 
     7 Circle of Commerce errors      negative 
     8 Circle of Commerce exquisite   positive 
     9 Circle of Commerce flourishing positive 
    10 Circle of Commerce eminent     positive 

Note: a few words in the Bing lexicon are labeled both positive and
negative, which is why `relationship = "many-to-many"` is set. Such a
word is counted once as positive and once as negative, so it has no
effect on net sentiment.

Checking how many positive and negative words were found in each
document.

``` r
tokens_sentiment %>%
  count(doc_title, sentiment)
```

    # A tibble: 6 × 3
      doc_title          sentiment     n
      <chr>              <chr>     <int>
    1 A06785             negative    680
    2 A06785             positive    703
    3 Circle of Commerce negative    422
    4 Circle of Commerce positive    489
    5 Free Trade         negative    281
    6 Free Trade         positive    344

### Step 3: Compute raw sentiment totals

Keeping only the two texts we are comparing (Circle of Commerce & Free
Trade), then collapsing to one row per document with positive, negative,
and net sentiment counts.

``` r
raw_sentiment <- tokens_sentiment %>%
  filter(doc_title %in% c("Circle of Commerce", "Free Trade")) %>%
  group_by(doc_title) %>%
  summarise(
    raw_positive = sum(sentiment == "positive"),
    raw_negative = sum(sentiment == "negative"),
    .groups = "drop"
  ) %>%
  mutate(raw_net = raw_positive - raw_negative)

# Preview: the raw sentiment summary table
raw_sentiment
```

    # A tibble: 2 × 4
      doc_title          raw_positive raw_negative raw_net
      <chr>                     <int>        <int>   <int>
    1 Circle of Commerce          489          422      67
    2 Free Trade                  344          281      63

**Observation:** Both texts have a small positive net score (67 for
Circle of Commerce and 63 for Free Trade). However, Circle of Commerce
is about 60% longer, so it naturally contains more sentiment words of
both kinds. Raw counts do not adjust for document length.

## Part II: TF-IDF-Weighted Sentiment Analysis

### Step 4: Compute TF-IDF for words in each document

Building the quanteda DFM from the same cleaned tokens used in Part I
(tokens_clean), so the preprocessing decisions are identical. All three
documents are included so that IDF has more information to work with.

``` r
# Grouping the cleaned words into one list per document
toks_list <- split(tokens_clean$word, tokens_clean$doc_title)

# Turn that list into a quanteda tokens object
toks <- as.tokens(toks_list)

# Build the document-feature matrix (rows = documents, columns = words)
dfm_mat <- dfm(toks)

# Preview: size of the DFM and the most frequent words overall
dfm_mat
```

    Document-feature matrix of: 3 documents, 7,668 features (54.75% sparse) and 0
    docvars.
                        features
    docs                 strong imagination nourished opinion suppose edward
      A06785                  3           3         1      13       5      7
      Circle of Commerce      0           1         0       1       3      2
      Free Trade              0           0         0       1       1      1
                        features
    docs                 misselden sallust allenger babylon
      A06785                    36       1        1       1
      Circle of Commerce         0       0        0       0
      Free Trade                 0       0        0       0
    [ reached max_nfeat ... 7,658 more features ]

``` r
top_words <- topfeatures(dfm_mat, 15)
tibble(word = names(top_words), count = top_words)
```

    # A tibble: 15 × 2
       word        count
       <chr>       <dbl>
     1 trade         618
     2 exchange      478
     3 merchants     308
     4 commodities   293
     5 money         290
     6 moneys        267
     7 haue          249
     8 hath          213
     9 kingdome      188
    10 cloth         180
    11 thereof       163
    12 malynes       162
    13 bee           159
    14 time          148
    15 price         146

Applying TF-IDF weighting to the DFM, using quanteda’s default settings.

``` r
dfm_tfidf_mat <- dfm_tfidf(dfm_mat)

# Preview: the TF-IDF matrix and the highest-weighted words overall
dfm_tfidf_mat
```

    Document-feature matrix of: 3 documents, 7,668 features (54.75% sparse) and 0
    docvars.
                        features
    docs                   strong imagination nourished opinion suppose edward
      A06785             1.431364   0.5282738 0.4771213       0       0      0
      Circle of Commerce 0          0.1760913 0               0       0      0
      Free Trade         0          0         0               0       0      0
                        features
    docs                 misselden   sallust  allenger   babylon
      A06785              17.17637 0.4771213 0.4771213 0.4771213
      Circle of Commerce   0       0         0         0        
      Free Trade           0       0         0         0        
    [ reached max_nfeat ... 7,658 more features ]

``` r
top_tfidf_words <- topfeatures(dfm_tfidf_mat, 15)
tibble(word = names(top_tfidf_words), tfidf = top_tfidf_words)
```

    # A tibble: 15 × 2
       word     tfidf
       <chr>    <dbl>
     1 moneys    47.0
     2 haue      43.8
     3 foreign   41.5
     4 kingdome  33.1
     5 ballance  32.4
     6 realm     30.5
     7 forraine  29.1
     8 malynes   28.5
     9 loss      28.2
    10 bee       28.0
    11 balance   24.8
    12 hee       23.1
    13 mony      22.2
    14 vpon      21.5
    15 native    19.1

### Step 5: Convert the TF-IDF results to a tidy table

Converting the TF-IDF DFM into a regular data frame. Reshaping it to one
row per document-word pair. Removing rows with zero weights. Keeping
only Circle of Commerce and Free Trade.

``` r
tfidf_tidy <- convert(dfm_tfidf_mat, to = "data.frame") %>%
  pivot_longer(
    cols      = -doc_id,
    names_to  = "word",
    values_to = "tfidf"
  ) %>%
  filter(tfidf > 0) %>%
  filter(doc_id %in% c("Circle of Commerce", "Free Trade")) %>%
  rename(doc_title = doc_id)

# Preview: the highest-weighted words, and how many words each document has left
tfidf_tidy %>%
  arrange(desc(tfidf)) %>%
  head(10)
```

    # A tibble: 10 × 3
       doc_title          word     tfidf
       <chr>              <chr>    <dbl>
     1 Circle of Commerce ballance  32.4
     2 Circle of Commerce forraine  29.1
     3 Circle of Commerce haue      28.7
     4 Circle of Commerce malynes   27.8
     5 Circle of Commerce mony      20.8
     6 Circle of Commerce hee       20.3
     7 Circle of Commerce kingdome  19.9
     8 Circle of Commerce bee       19.0
     9 Free Trade         haue      15.1
    10 Free Trade         kingdome  13.2

``` r
tfidf_tidy %>% count(doc_title)
```

    # A tibble: 2 × 2
      doc_title              n
      <chr>              <int>
    1 Circle of Commerce  3325
    2 Free Trade          2246

### Step 6: Keep only sentiment-bearing words

Joining the TF-IDF table to the Bing lexicon. This filters the existing
TF-IDF results to sentiment words only. TF-IDF is not recalculated.

``` r
tfidf_sentiment <- tfidf_tidy %>%
  inner_join(bing, by = "word", relationship = "many-to-many")

# Preview: how many sentiment words remain in each document
tfidf_sentiment %>% count(doc_title, sentiment)
```

    # A tibble: 4 × 3
      doc_title          sentiment     n
      <chr>              <chr>     <int>
    1 Circle of Commerce negative    164
    2 Circle of Commerce positive    108
    3 Free Trade         negative     90
    4 Free Trade         positive     74

Inspecting the sentiment-bearing words with the highest TF-IDF weights
in each document, separately for positive and negative words. These are
the words that drive the TF-IDF-weighted sentiment scores.

``` r
top_sentiment_words <- tfidf_sentiment %>%
  group_by(doc_title, sentiment) %>%
  slice_max(tfidf, n = 8, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(doc_title, sentiment, desc(tfidf))

top_sentiment_words %>% print(n = Inf)
```

    # A tibble: 32 × 4
       doc_title          word        tfidf sentiment
       <chr>              <chr>       <dbl> <chr>    
     1 Circle of Commerce impertinent 2.86  negative 
     2 Circle of Commerce imperfect   2.39  negative 
     3 Circle of Commerce meddle      1.91  negative 
     4 Circle of Commerce kill        1.43  negative 
     5 Circle of Commerce scandalous  1.43  negative 
     6 Circle of Commerce wrong       1.43  negative 
     7 Circle of Commerce forbid      1.43  negative 
     8 Circle of Commerce strictly    1.43  negative 
     9 Circle of Commerce passion     4.29  positive 
    10 Circle of Commerce divine      3.34  positive 
    11 Circle of Commerce affection   2.86  positive 
    12 Circle of Commerce humane      1.91  positive 
    13 Circle of Commerce holy        1.43  positive 
    14 Circle of Commerce wise        1.06  positive 
    15 Circle of Commerce favour      1.06  positive 
    16 Circle of Commerce grace       1.06  positive 
    17 Free Trade         imposition  3.34  negative 
    18 Free Trade         disorderly  2.86  negative 
    19 Free Trade         deficient   1.43  negative 
    20 Free Trade         pleas       1.43  negative 
    21 Free Trade         loose       1.41  negative 
    22 Free Trade         cruelty     0.954 negative 
    23 Free Trade         extortion   0.954 negative 
    24 Free Trade         repugnant   0.954 negative 
    25 Free Trade         peace       1.23  positive 
    26 Free Trade         happy       1.23  positive 
    27 Free Trade         richly      0.954 positive 
    28 Free Trade         wonderfully 0.954 positive 
    29 Free Trade         sincerity   0.954 positive 
    30 Free Trade         suffice     0.954 positive 
    31 Free Trade         protect     0.954 positive 
    32 Free Trade         mightily    0.954 positive 

**Observation:** After TF-IDF weighting, many of the highest-weighted
sentiment words are not about trade (for example “divine”, “passion”,
“holy” in Circle of Commerce). Some trade terms also appear to be
misread by the modern Bing lexicon, for example “imposition” in Free
Trade, which in this period could refer to a duty on goods.

### Step 7: Compute TF-IDF-weighted sentiment totals

Summing the TF-IDF weights (not counting words) of positive and negative
words for each document, then calculating net TF-IDF sentiment.

``` r
tfidf_sentiment_summary <- tfidf_sentiment %>%
  group_by(doc_title, sentiment) %>%
  summarise(total_tfidf = sum(tfidf), .groups = "drop") %>%
  pivot_wider(
    names_from  = sentiment,
    values_from = total_tfidf,
    values_fill = 0
  ) %>%
  rename(
    tfidf_positive = positive,
    tfidf_negative = negative
  ) %>%
  mutate(tfidf_net = tfidf_positive - tfidf_negative)

# Preview: the TF-IDF sentiment summary table
tfidf_sentiment_summary
```

    # A tibble: 2 × 4
      doc_title          tfidf_negative tfidf_positive tfidf_net
      <chr>                       <dbl>          <dbl>     <dbl>
    1 Circle of Commerce           86.2           62.9     -23.3
    2 Free Trade                   44.2           32.7     -11.5

## Part III: Compare Raw vs. TF-IDF Sentiment

Combining the raw-count and TF-IDF-weighted summaries into one table
(one row per document), and exporting it as a .csv file.

``` r
sentiment_comparison <- raw_sentiment %>%
  left_join(tfidf_sentiment_summary, by = "doc_title") %>%
  select(
    doc_title,
    raw_positive, raw_negative, raw_net,
    tfidf_positive, tfidf_negative, tfidf_net
  ) %>%
  mutate(across(starts_with("tfidf"), ~ round(.x, 3)))

# Preview: the final comparison table
sentiment_comparison %>% print(width = Inf)
```

    # A tibble: 2 × 7
      doc_title          raw_positive raw_negative raw_net tfidf_positive
      <chr>                     <int>        <int>   <int>          <dbl>
    1 Circle of Commerce          489          422      67           62.9
    2 Free Trade                  344          281      63           32.7
      tfidf_negative tfidf_net
               <dbl>     <dbl>
    1           86.2     -23.3
    2           44.2     -11.5

``` r
# Save the table as a .csv file in the same folder as this notebook
write_csv(sentiment_comparison, "sentiment_comparison.csv")
```

### Which words drove the change?

Finding the most frequent sentiment words in Part I that received a
TF-IDF weight of zero in Part II (because they appear in all three
texts). These words counted in the raw method but disappeared in the
TF-IDF method.

``` r
dropped_words <- tokens_sentiment %>%
  filter(doc_title %in% c("Circle of Commerce", "Free Trade")) %>%
  count(doc_title, word, sentiment, name = "raw_count") %>%
  anti_join(tfidf_sentiment, by = c("doc_title", "word")) %>%
  group_by(doc_title, sentiment) %>%
  slice_max(raw_count, n = 8, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(doc_title, sentiment, desc(raw_count))

dropped_words %>% print(n = Inf)
```

    # A tibble: 32 × 4
       doc_title          word      sentiment raw_count
       <chr>              <chr>     <chr>         <int>
     1 Circle of Commerce decay     negative         14
     2 Circle of Commerce stranger  negative         13
     3 Circle of Commerce abuse     negative         12
     4 Circle of Commerce fall      negative         11
     5 Circle of Commerce false     negative         10
     6 Circle of Commerce mystery   negative          9
     7 Circle of Commerce lost      negative          8
     8 Circle of Commerce fallen    negative          7
     9 Circle of Commerce free      positive         29
    10 Circle of Commerce treasure  positive         18
    11 Circle of Commerce benefit   positive         14
    12 Circle of Commerce noble     positive         14
    13 Circle of Commerce respect   positive         14
    14 Circle of Commerce worthy    positive         14
    15 Circle of Commerce excellent positive         13
    16 Circle of Commerce worth     positive         13
    17 Free Trade         decay     negative         27
    18 Free Trade         false     negative         15
    19 Free Trade         abuse     negative         12
    20 Free Trade         strict    negative          7
    21 Free Trade         fallen    negative          6
    22 Free Trade         lost      negative          6
    23 Free Trade         abused    negative          5
    24 Free Trade         fall      negative          5
    25 Free Trade         remedy    positive         40
    26 Free Trade         treasure  positive         24
    27 Free Trade         liberty   positive         19
    28 Free Trade         benefit   positive         18
    29 Free Trade         respect   positive         10
    30 Free Trade         pleasure  positive          9
    31 Free Trade         ready     positive          8
    32 Free Trade         worthy    positive          7

1.  What changed between the two methods?

The direction of the sentiment flipped. With raw counts, both texts came
out slightly positive: Circle of Commerce at +67 (489 positive, 422
negative) and Free Trade at +63 (344 positive, 281 negative). Going by
that alone, I would have said the two texts have almost the same tone.
Once I weighted the words by TF-IDF, both texts turned negative. Circle
of Commerce came to −23.31 and Free Trade at −11.50. The gap between
them also got bigger. Under TF-IDF, Circle of Commerce carries about
twice as much net negative weight as Free Trade, while under raw counts
they were nearly equal.

2.  Why did TF-IDF alter the results?

TF-IDF gives zero weight to any word that shows up in all three texts,
and rewards words that only show up in one or two. That turned out to
matter a lot here, because many of the most common positive words are
shared across all three documents. Words like “free” (29 times in Circle
of Commerce), “remedy” (40 times in Free Trade) and “treasure” counted
heavily in the raw method, but got a weight of zero under TF-IDF. Once
those were gone, what was left were more rare words, and more of them
were negative (164 negative vs 108 positive distinct words in Circle of
Commerce, and 90 vs 74 in Free Trade). This does not mean the texts are
negative. It means that the sentiment words each text uses on its own
are mostly negative, while the sentiment words all three texts share are
mostly positive. Also, with only three documents, IDF can only take
three values. So a word used just a few times in one text can still get
a high weight.

3.  Which specific words drove the changes?

I see two groups. The first is the frequent positive words that all
three texts share and that TF-IDF wiped out: “free”, “treasure”,
“benefit”, “remedy” and “liberty”. Several of these probably belong to
the economic vocabulary of the debate rather than expressing any real
feeling, which makes me question how much they should have counted in
the raw score to begin with. The second group is the rare words that
gained weight. In Circle of Commerce, the top negative words were
“impertinent”, “imperfect” and “meddle”, and the top positive ones were
“passion”, “divine” and “affection”. In Free Trade, the top negative
words were “imposition” and “disorderly”, and the top positive ones were
“peace” and “happy”. A lot of these have nothing to do with trade. Some
also look misread by the Bing lexicon. “Imposition”, for example, could
mean a duty on goods in the 1620s, but Bing treats it as negative word.

4.  When would each method be more appropriate? What limitation does
    document length create for the raw-count measure?

Raw counts are better when I want to know how much positive and negative
language a text uses in total, including the common words it shares with
the other texts. TF-IDF makes more sense when I want to know what is
distinctive about one text’s sentiment compared to the others, for
example how Misselden’s tone differs from his rivals. I would not use
TF-IDF to say how positive or negative one particular document is, and
its results depend heavily on which documents are in the corpus. The
problem with raw counts is length. A longer text will simply have more
positive and more negative words. After cleaning, Circle of Commerce has
12,972 words and Free Trade has 8,024. Their raw net scores look almost
the same (+67 vs +63), but per 1,000 words, Free Trade is actually more
positive (about 7.9 vs 5.2). Dividing by document length would make the
raw comparison fairer.
