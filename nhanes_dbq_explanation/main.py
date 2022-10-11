
from analyze_nhanes_data import NhanesDataAnalyzer


if __name__ == '__main__':

    random_seed = 2022
    nhanes_data = {'2007-2008': ['DEMO_E.XPT', 'DBQ_E.XPT'],
                   '2009-2010': ['DEMO_F.XPT', 'DBQ_F.XPT'],
                   '2011-2012': ['DEMO_G.XPT', 'DBQ_G.XPT'],
                   '2013-2014': ['DEMO_H.XPT', 'DBQ_H.XPT'],
                   '2015-2016': ['DEMO_I.XPT', 'DBQ_I.XPT'],
                   '2017-2018': ['DEMO_J.XPT', 'DBQ_J.XPT'],}

    nda = NhanesDataAnalyzer(nhanes_data, random_seed)
    nda.analyze()

