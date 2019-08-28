# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


def normalize(x):
    return x / 2 - 1

def inv_normalize(x):
    return (x + 1) / 2