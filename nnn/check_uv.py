import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns

# from magicgui import magicgui
# from magicgui import widgets
# import pathlib

# from . import uv

sns.set_style('ticks')
sns.set_context('paper')

# from magicgui.experimental import guiclass, button

@guiclass
class MyDataclass:
    a: int = 0
    b: str = 'hello'
    c: bool = True

    @button
    def compute(self):
        print(self.a, self.b, self.c)

obj = MyDataclass(a=10, b='foo')
obj.gui.show()

# parser = argparse.ArgumentParser(description='Launch a GUI to manually annotate')

# parser.add_argument('-s', '--samplesheet', required=True, help='filename of the sampleheet csv file')
# parser.add_argument('-f', '--figdir', help='figure dir where the fitted images are')

# class AnnotationGUI(object):

#     def __init__(self,
#         samplesheet, figdir
#         ):
        
#         self.figdir = figdir
#         self.sample_df = uv.read_sample_sheet(samplesheet)



# if __name__ == "__main__":

#     args = parser.parse_args()

#     gui = AnnotationGUI(
#         samplesheet = args.samplesheet,
#         figdir = args.figdir
#         )

#     gui.run()