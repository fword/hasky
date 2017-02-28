from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys  
  
current_word = None  
current_count = 0  
word = None  
  
min_count = 0
if len(sys.argv) > 1:
  min_count = int(sys.argv[1])

for line in sys.stdin:  
  line = line.rstrip()  
  try:
    word, count = line.split('\t', 1)  
  except Exception:
    print('bad line split:', line, file=sys.stderr)
    continue
  
  try:  
    count = int(count)  
  except ValueError:  
    print('bad line not int count:', line, file=sys.stderr)
    continue  
 
  if current_word == word:  
    current_count += count  
  else:  
    if current_word:  
      if not min_count or current_count >= min_count:
        print('%s\t%d' % (current_word, current_count))
    current_count = count  
    current_word = word  
  
if current_word == word:  
  if not min_count or current_count >= min_count:
    print('%s\t%d' % (current_word, current_count))