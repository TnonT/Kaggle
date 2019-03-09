# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-2-22下午2:24
# @file:re_learn.py

import re

# s = "JD is a ab h  s asd."
#
# print(re.sub(r'\s+', '-', s))
#
# x = "i didn't care about you."
# mispell_dict = {
#     "didn't" : "did not",
#     'counselling' : 'counseling',
#     'theatre' : 'theater',
#     'cancelled' : 'canceled'
# }
#
# # print('|'.join(mispell_dict.keys()))
# mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
#
#
# def replace(match):
#     print("===========replace=========")
#     print(match.group(0))
#     return mispell_dict[match.group(0)]
#
#
# x = mispell_re.sub(replace, x)
#
# print(x)
#
# y = mispell_re.match(x)
# print(y)

m = re.search('fo', 'fseefoaod onm the')
print(m.group())