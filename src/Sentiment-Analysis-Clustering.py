'''
Author: Yuhao Jia 
Created: 02/20/2022

coding: utf-8 
'''
from bs4 import BeautifulSoup
from kneed import KneeLocator
import math
from matplotlib import pyplot as plt
import requests
import pandas as pd
from requests_html import HTMLSession
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob as tb

url = 'https://finance.yahoo.com/most-active?count=100&offset=0'
headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"}
r = requests.get(url, headers = headers)

soup = BeautifulSoup(r.content, 'html.parser')

_dictofs = {}

for item in soup.select('.simpTblRow'):
    c = item.select('[aria-label=Name]')[0].get_text()
    pc = item.select('[aria-label="% Change"]')[0].get_text()
    _dictofs[c] = pc[:-1]

def avgsentiment(company):

    s = HTMLSession()
    url = f'https://news.google.com/rss/search?q={company}'

    results = s.get(url)

    _lot = []
    _dictofsent = {}

    for title in results.html.find('title'):
        _lot.append(title.text)

    _lot = _lot[1:]

    for title in _lot:
        b = tb(title)
        _dictofsent[b.sentiment.polarity] = b.sentiment.subjectivity

    counter = 0
    sent = 0

    for key in _dictofsent:
        if _dictofsent[key] < 0.5:
            counter += 1
            sent += _dictofsent[key]
    try:
        avg = sent / counter
    except:
        avg = sent
    else:
        avg = sent / counter
    
    return(avg)

d = {'Company': [], 'Avg Sentiment Score': [], '% Change': []}

for key in _dictofs:
    a = avgsentiment(key)
    d['Company'].append(key)
    d['Avg Sentiment Score'].append(a)
    d['% Change'].append(_dictofs[key])

df = pd.DataFrame(data=d)

k_range = range(1,10)
sse = []
for k in k_range:
    km = KMeans(n_clusters = k)
    km.fit(df[['Avg Sentiment Score','% Change']])
    sse.append(km.inertia_)

sse_r = range(1,len(sse)+1)

nclu = KneeLocator(sse_r, sse, curve = 'convex', direction = 'decreasing').knee

km = KMeans(n_clusters = nclu)
_clustd = km.fit_predict(df[['Avg Sentiment Score','% Change']]) 
df['cluster'] = _clustd

scal = MinMaxScaler()

scal.fit(df[['Avg Sentiment Score']])
df['Avg Sentiment Score'] = scal.transform(df[['Avg Sentiment Score']])

scal.fit(df[['% Change']])
df['% Change'] = scal.transform(df[['% Change']])

df_is = {}
for i in range (0,nclu):
    df_is[f"_df{i}"] = df[df.cluster == i]

_locolour = ["Red", "Black", "Orange", "Green", "Blue", "Purple"]
col_count = 0

## The following class definition is to interact with the resulting plot, credit is given as follows:
## AndrewStraw, GaelVaroquaux, Unknown[99], AngusMcMorland, newacct, danielboone(2016)Matplotlib: interactive plotting[Source code]. https://scipy-cookbook.readthedocs.io/items/Matplotlib_Interactive_Plotting.html

class AnnoteFinder(object):
    def __init__(self, xdata, ydata, annotes, ax=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def __call__(self, event):

        if event.inaxes:

            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                # print(event.xdata, event.ydata)
                for x, y, a in self.data:
                    # print(x, y, a)
                    if ((clickX-self.xtol < x < clickX+self.xtol) and
                            (clickY-self.ytol < y < clickY+self.ytol)):
                        annotes.append(
                            (self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, ax, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            t = ax.text(x, y, " - %s" % (annote),)
            m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)

plt.rcParams["font.family"] = "sans-serif"
            
fig, ax = plt.subplots()
ax.set_title("Avg Sentiment Score vs % Change for the Most Active Stocks Today")
ax.set_xlabel("Average Sentiment Score")
ax.set_ylabel("% Change")

for key in df_is:
    x = df_is[key]['Avg Sentiment Score']
    y = df_is[key]['% Change']
    ax.scatter(x,y,color = _locolour[col_count])
    af = AnnoteFinder(x,y,df_is[key]['Company'], ax = ax)
    fig.canvas.mpl_connect('button_press_event', af)
    col_count += 1

plt.show()
