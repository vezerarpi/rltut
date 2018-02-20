'''Choose and cache an IPython password.
'''

import os
import sys
import json
import IPython.lib


PASSWORD_FILE = '.notebook-password'

if not os.path.isfile(PASSWORD_FILE):
    sys.stderr.write('Enter a new notebook password: ')
    password = input()
    hash_ = IPython.lib.passwd(password)
    with open(PASSWORD_FILE, 'w') as f:
        json.dump(dict(password=password, hash=hash_), f)

with open(PASSWORD_FILE) as f:
    p = json.load(f)
    sys.stderr.write('Your password is: {}\n'.format(p['password']))
    sys.stdout.write('{}\n'.format(p['hash']))
