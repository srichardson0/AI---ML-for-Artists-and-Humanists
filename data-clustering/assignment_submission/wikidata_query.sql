# Wikidata SPARQL query: background attributes for everyone with a U.S. Congress BioGuide ID
# Run at https://query.wikidata.org/ (download as CSV) or from the notebook with REFRESH_WIKIDATA = True.
#
# Design notes
# - One row per person. Multi-valued properties are packed into one cell with "|" between values,
#   which avoids the row explosion (3 jobs x 2 degrees x 4 offices = 24 rows) that made the first
#   version of this query time out.
# - No position-held filter. The join with the CEL House file in Python does the filtering,
#   so only people who actually served in the House in 2001-2024 are kept.
# - Birth year >= 1920 just trims 19th- and early-20th-century members to keep the query fast.

SELECT ?bioguideId ?personLabel
       (GROUP_CONCAT(DISTINCT ?occLabel; separator="|") AS ?occupations)       # P106 occupation
       (GROUP_CONCAT(DISTINCT ?degLabel; separator="|") AS ?degrees)           # P512 academic degree
       (GROUP_CONCAT(DISTINCT ?milLabel; separator="|") AS ?militaryBranches)  # P241 military branch
WHERE {
  ?person wdt:P1157 ?bioguideId ;          # has a BioGuide ID (the join key)
          wdt:P569 ?birth .
  FILTER(YEAR(?birth) >= 1920)

  OPTIONAL { ?person wdt:P106 ?occ . FILTER(?occ != wd:Q82955)   # skip the generic "politician"
             ?occ rdfs:label ?occLabel . FILTER(LANG(?occLabel) = "en") }
  OPTIONAL { ?person wdt:P512 ?deg .
             ?deg rdfs:label ?degLabel . FILTER(LANG(?degLabel) = "en") }
  OPTIONAL { ?person wdt:P241 ?mil .
             ?mil rdfs:label ?milLabel . FILTER(LANG(?milLabel) = "en") }
  OPTIONAL { ?person rdfs:label ?personLabel . FILTER(LANG(?personLabel) = "en") }
}
GROUP BY ?bioguideId ?personLabel
