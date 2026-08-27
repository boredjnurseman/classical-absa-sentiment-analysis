# Hu-Liu review data

The raw review corpora are not redistributed in this repository. Download the
archives from Bing Liu's University of Illinois Chicago project page and place
the extracted text files in this layout:

```text
data/raw/
    Customer_review_data/
    Reviews-9-products/
    CustomerReviews-3_domains/
```

Official archives:

- [Customer Review Data (five products)](https://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip)
- [Additional Reviews (nine products)](https://www.cs.uic.edu/~liub/FBS/Reviews-9-products.rar)
- [Customer Reviews (three domains)](https://www.cs.uic.edu/~liub/FBS/CustomerReviews-3-domains.rar)

The archives use two directory names that differ from the package's canonical
layout. After extraction, rename `customer review data` to
`Customer_review_data`, and rename `CustomerReviews -3 domains (IJCAI2015)` to
`CustomerReviews-3_domains`. Keep `Reviews-9-products` unchanged. The parser
accepts the title-delimited files and the line-oriented iPod file within that
layout.

The full run also requires spaCy's `en_core_web_sm` model at version 3.8.0.
The exact installation command is documented in the project README.

The repository's MIT licence covers its code and documentation only. It does
not assign a licence to these separately distributed corpora. Cite the source
works requested by the dataset authors, including Hu and Liu (2004), when using
the data.
