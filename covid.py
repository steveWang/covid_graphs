from collections import defaultdict
from datetime import timedelta
from matplotlib.dates import datestr2num, num2date, DateFormatter

import csv
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import requests


matplotlib.use('gtk3agg')
MEAN_WINDOW=7
# States to omit in US total
EXCLUSIONS = {}

states_of_interest = ['D', 'R', 'US', 'Dc', 'Rc']
states_of_interest = ['MA', 'PA', 'IL', 'US']
# states_of_interest = ['ME', 'MA', 'NY']
# states_of_interest = ['US']
# states_of_interest = ['AZ', 'FL', 'TX', 'NY', 'US']

# States with > 5% margin in the 2016 election.
ds = {'DC', 'HI', 'MD', 'MA', 'NY', 'VT', 'IL', 'NJ', 'CT', 'RI', 'OR', 'NM', 'CA'}
rs = {'NE', 'WY', 'WV', 'OK', 'ND', 'ID', 'KY', 'SD', 'AL', 'AR', 'TN', 'KS', 'MT', 'LA', 'MO', 'UT', 'MS', 'AK', 'SC', 'IA', 'TX', 'OH'}

# Counties with > 10% margin in the 2016 election.
counties = {'Dc': set(), 'Rc': set()}

START_DATE = num2date(datestr2num('2020-05-01'))

state_aliases = {
  'Alabama': 'AL',
  'Alaska': 'AK',
  'Arizona': 'AZ',
  'Arkansas': 'AR',
  'California': 'CA',
  'Colorado': 'CO',
  'Connecticut': 'CT',
  'Delaware': 'DE',
  'District of Columbia': 'DC',
  'Florida': 'FL',
  'Georgia': 'GA',
  'Guam': 'GU',
  'Hawaii': 'HI',
  'Idaho': 'ID',
  'Illinois': 'IL',
  'Indiana': 'IN',
  'Iowa': 'IA',
  'Kansas': 'KS',
  'Kentucky': 'KY',
  'Louisiana': 'LA',
  'Maine': 'ME',
  'Maryland': 'MD',
  'Massachusetts': 'MA',
  'Michigan': 'MI',
  'Minnesota': 'MN',
  'Mississippi': 'MS',
  'Missouri': 'MO',
  'Montana': 'MT',
  'Nebraska': 'NE',
  'Nevada': 'NV',
  'New Hampshire': 'NH',
  'New Jersey': 'NJ',
  'New Mexico': 'NM',
  'New York': 'NY',
  'North Carolina': 'NC',
  'North Dakota': 'ND',
  'Northern Mariana Islands': 'MP',
  'Ohio': 'OH',
  'Oklahoma': 'OK',
  'Oregon': 'OR',
  'Pennsylvania': 'PA',
  'Puerto Rico': 'PR',
  'Rhode Island': 'RI',
  'South Carolina': 'SC',
  'South Dakota': 'SD',
  'Tennessee': 'TN',
  'Texas': 'TX',
  'Utah': 'UT',
  'Vermont': 'VT',
  'Virginia': 'VA',
  'Virgin Islands': 'VI',
  'Washington': 'WA',
  'West Virginia': 'WV',
  'Wisconsin': 'WI',
  'Wyoming': 'WY',
  'US': 'US',
}

def populate_counties():
  PATH = ('US_County_Level_Election_Results_08-16/'
          '2016_US_County_Level_Presidential_Results.csv')
  THRESHOLD = 0.1
  dcount, rcount = 0, 0
  with open(PATH) as csvfile:
    for line in csv.DictReader(csvfile):
      margin = float(line['per_dem']) - float(line['per_gop'])
      fips = line['combined_fips']
      if margin > THRESHOLD:
        counties['Dc'].add(fips)
        dcount += float(line['total_votes'])
      if margin < -THRESHOLD:
        counties['Rc'].add(fips)
        rcount += float(line['total_votes'])
  print(dcount, rcount)

def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_geo_mean(x, N):
  logs = np.log(x)
  return math.e ** np.array(list(sum(logs[i:i+N]) / N for i in range(len(x) - N+1)))

def running_median(x, N):
  return [sorted(x[i:i+N])[N//2] for i in range(len(x) - N + 1)]

def difference(ys):
  return [ys[n+1] - ys[n] for n in range(len(ys) - 1)]

def ratio(ys):
  return [ys[n+1] / max(1, ys[n]) for n in range(len(ys) - 1)]

def graph_cases(xs, cases, finalize=True, ts=0, epoch=START_DATE):
  ax = plt.gca()
  ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
  to_skip = len([x for x in xs if x < epoch])
  plt.xlim(left=max(epoch, xs[0]), right=xs[-1])
  for state in states_of_interest:
    if state not in cases:
      continue
    if isinstance(cases[state], dict):
      ys = [cases[state].get(x, 0) for x in xs]
    else:
      ys = cases[state]

    ys = difference(ys)
    ys = ys[max(to_skip-MEAN_WINDOW, 0):]
    zs = time_shift(xs, ts)[-len(ys):]
    color=next(ax._get_lines.prop_cycler)['color']
    plt.plot(zs, ys, linestyle=':', color=color)
    plt.plot(zs[MEAN_WINDOW-1:], running_mean(ys, MEAN_WINDOW),
      label=state, color=color)
  if finalize:
    finalize_plot()
  else:
    ax.set_prop_cycle(None)

def finalize_plot():
  plt.legend()
  plt.title('New infections')
  plt.ylabel('cases')
  plt.yscale('log')
  plt.show()

def time_shift(xs, shift):
  return [x + timedelta(shift) for x in xs]

def canonicalize(state):
  state = state.split(', ')[-1]
  if state in state_aliases:
    return state_aliases[state]
  return state

def add_date(cases, state, date, count):
  if state not in EXCLUSIONS:
    cases['US'][date] = cases['US'].get(date, 0) + count
  if state in ds:
    cases['D'][date] = cases['D'].get(date, 0) + count
  if state in rs:
    cases['R'][date] = cases['R'].get(date, 0) + count
  cases[state][date] = cases[state].get(date, 0) + count

def build_from_jhu_archive():
  PATH = ('jhu/archived_data/archived_time_series/'
          'time_series_19-covid-Confirmed_archived_0325.csv')

  cases = defaultdict(dict)
  axis = []

  with open(PATH) as csvfile:
    for line in csv.reader(csvfile):
      if not len(axis):
        axis = [num2date(datestr2num(x)) for x in line[4:]]
      state, country, lat, lng = line[:4]
      if country != 'US':
        continue
      state = canonicalize(state)

      dates = [int(x or 0) for x in line[4:]]
      for i, date in enumerate(axis):
        add_date(cases, state, date, dates[i])

  plt.plot(axis, [max(1, 2e-4 * 10 ** (x/8)) for x in range(len(axis))],
           label='10x growth every 8 days')
  graph_cases(axis, cases)

def build_from_jhu_reports():
  GLOB_PATH = 'jhu/csse_covid_19_data/csse_covid_19_daily_reports/*.csv'
  xs = []
  cases = defaultdict(dict)
  for f in sorted(glob.glob(GLOB_PATH)):
    with open(f) as csvfile:
      date = num2date(datestr2num(os.path.basename(f).strip('.csv')))
      xs.append(date)

      reader = csv.DictReader(csvfile)
      country_field = 'Country/Region'
      state_field = '\ufeffProvince/State'
        
      if country_field not in reader.fieldnames:
        country_field = 'Country_Region'

      if state_field not in reader.fieldnames:
        state_field = 'Province/State'
      if state_field not in reader.fieldnames:
        state_field = 'Province_State'

      cases['US'][date] = 0
      for line in reader:
        country, state = line[country_field], canonicalize(line[state_field])
        if country != 'US':
          continue
        if state == 'Recovered':
          continue
        add_date(cases, state, date, int(line['Confirmed'] or 0))

  # plt.plot(xs, [max(700, 700 * 10 ** ((x - 132) / 30)) for x in range(len(xs))],
  #          label='10x growth every 30 days', linestyle='-.')
  # plt.plot(xs, [max(80000, 15000 * 10 ** ((x - 1) / 150)) for x in range(len(xs))],
  #          label='10x growth every 150 days', linestyle='-.')
  ys = [cases[state].get(date, 0) for date in xs]
  graph_cases(xs, cases)

def build_from_nyt():
  PATH = '/home/steve/workspace/scratch/covid_data/nyt/us-states.csv'
  axis = set()
  cases = defaultdict(dict)
  deaths = defaultdict(dict)
  with open(PATH) as csvfile:
      reader = csv.DictReader(csvfile)
      for line in reader:
        date = num2date(datestr2num(line['date']))
        axis.add(date)
        state = canonicalize(line['state'])
        c, d = int(line['cases']), int(line['deaths'])
        add_date(cases, state, date, c)
        add_date(deaths, state, date, d)

  xs = sorted(axis)
  graph_cases(xs, cases)

def build_from_nyt_counties():
  populate_counties()
  PATH = '/home/steve/workspace/scratch/covid_data/nyt/us-counties.csv'
  axis = set()
  cases = defaultdict(dict)
  deaths = defaultdict(dict)
  with open(PATH) as csvfile:
      reader = csv.DictReader(csvfile)
      for line in reader:
        date = num2date(datestr2num(line['date']))
        axis.add(date)
        state = canonicalize(line['state'])
        fips = line['fips']
        c, d = int(line['cases']), int(line['deaths'])
        for k,s in counties.items():
          if fips in s:
            cases[k][date] = cases[k].get(date, 0) + c
            deaths[k][date] = deaths[k].get(date, 0) + d
        add_date(cases, state, date, c)
        add_date(deaths, state, date, d)

  xs = sorted(axis)
  graph_cases(xs, cases)

def build_from_covid_tracking():
  req = requests.get('https://covidtracking.com/api/v1/states/daily.csv')
  if not req.ok:
    return
  reader = csv.DictReader(req.text.split('\n'))
  axis = set()
  cases = defaultdict(dict)
  deaths = defaultdict(dict)
  for line in reader:
    date = num2date(datestr2num(line['date']))
    axis.add(date)
    state = line['state']
    d = int(line['death'] or 0)
    p = int(line['positive'] or 0)
    n = int(line['negative'] or 0) + int(line['pending'] or 0)

    state = canonicalize(state)
    if state in state_aliases:
      state = state_aliases[state]

    add_date(cases, state, date, p)
    add_date(deaths, state, date, d)
  xs = sorted(axis)
  graph_cases(xs, cases)

# build_from_jhu_archive()
# build_from_jhu_reports()
# build_from_covid_tracking()
build_from_nyt()
# build_from_nyt_counties()
