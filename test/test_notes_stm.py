# coding: utf8


# ppt algorithm
#    procedure:
#   (1) set std
#   (2) set points
#   (3) get out points ratio(cdf), using strategy
#   (4) get raw score percent and cumulative percent
#   (5) create map_table: map raw point to out point, using strategies
#   (6) transform raw score to out score for each record(person)
#   strategy:
#   (1) set point ratio strategy:
#        1. add trunc error at min point: yes, no
#        2. add trunc error at max point: yes, no
#        3. define section at each point: left, right, middle
