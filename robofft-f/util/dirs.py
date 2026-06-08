# MIT License

# Copyright (c) 2025 RoboFFT-F Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
ROBOFFT_F_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not ROBOFFT_F_DIR == os.environ['ROBOFFT_F_DIR']:
    raise ValueError(f"Hey did you correctly set up your env variable ROBOFFT_F_DIR? It shows that ROBOFFT_F_DIR={os.environ['ROBOFFT_F_DIR']} but the code rest in {ROBOFFT_F_DIR}. ")

ROBOFFT_F_CFG_DIR = os.path.join(ROBOFFT_F_DIR, 'cfg')

ROBOFFT_F_DATA_DIR = os.environ['ROBOFFT_F_DATA_DIR']

ROBOFFT_F_LOG_DIR =  os.environ['ROBOFFT_F_LOG_DIR']
