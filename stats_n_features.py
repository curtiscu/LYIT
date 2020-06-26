"""
Code for collecting basic stats and features
about a DataFrame of MIDI file data. The expectation
is the DataFrame passed in bas been loaded 
using function 'data_prep.load_file()'

"""

# imports
import pandas as pd
import numpy as np
import datetime
import data_prep as dp
from data_prep import MidiTools as mt


def frequency_table(midi_file_data):
  '''Generate histogram/ frequency tablet of hits
  
  NOTE: assumes parameter 'midi_file_data' has been 
  created/ formatted by loading a MIDI file
  using function 'data_prep.load_file()'
  '''
  tmp_df = midi_file_data.copy()
  tmp2 = tmp_df.drop(columns=['bar_beat_number', 'note'])

  # calculate number of hits per instrument, per metric position 
  op2 = tmp2.groupby(by=['bar_beat_number', 'note'])['velocity'].count()
  op3 = pd.DataFrame(op2)
  op3.columns = ['hit_count']
  
  # flip things around, fill with '0' instead of NaN, then convert to int
  op4 = op3.unstack(0).fillna(0).astype('int32')

  # Flesh out df with any missing quantize buckets
  op4.columns = op4.columns.droplevel(level=0) 
  fill_cols = pd.DataFrame(columns=[x for x in range(1, 17)])
  full_col_list = list(set().union(op4.columns, fill_cols.columns))
  op5 = op4.reindex(columns=full_col_list, fill_value=0)

  # Label the instruments in the df
  mt.getInstruments(op5.index.values)
  new_row_index = dict(zip(op5.index.values, mt.getInstruments(op5.index.values)))
  op6 = op5.rename(new_row_index)

  # DEBUG: Review completed Frequency Table
  #display(op6)
  
  return op6


# TODO: doesn't take action, just informational 
#       for now. need to decide what to do with
#       the information if this returns dataframe 
#       with df.size > 0 
def error_buckets(midi_file_data):
  '''Detect multiple hits in quantize bucket.
  
  Detect problem beats in bars that have more hits
  at a quantize level more detailed than the code 
  can handle, i.e. below 16th note level.
  
  NOTE..
  - assumes parameter 'midi_file_data' has been 
  created/ formatted by loading a MIDI file
  using function 'data_prep.load_file()'
  
  - this doesn't actually take any action such as
  filtering, just for informational purposes
  '''
  m_hits = midi_file_data.copy()
  m_hits.drop(columns=['note'], inplace=True)

  # filters entire dataframe to only show groups
  # with more than 1 member, i.e. duplicate htis in bucket
  # Basically the output below must (currently) be flagged as an error
  result = m_hits.groupby(['beat_center','note']).filter(lambda a_group: len(a_group) > 1)
  
  return result


def gather_stats(midi_file_data):
  '''Returns dictionary, keys are instruments, values
  are stats dataframes with averages, standard deviations, IQR
  min/ max, etc..
  
  NOTE..
  - assumes parameter 'midi_file_data' has been 
  created/ formatted by loading a MIDI file
  using function 'data_prep.load_file()'
  '''
  stats_df = midi_file_data.drop(columns=['bar_beat_number', 'note', 'track_msg_num', 'total_ticks', 'beat_center', 'bar_number' ]).copy()
  
  stats_1 = stats_df.groupby(by=['bar_beat_number', 'note']).agg(
    hits=('velocity', "count"),               # count of hits at position
    off_min=('beat_offset', "min"),           # earliest hit offset time
    off_max=('beat_offset', "max"),           # latest hit offset time 
    off_median =('beat_offset', np.median),   # median offset
    off_mean=('beat_offset', np.mean),        # mean offset
    off_iqr=('beat_offset', iqr),             # IQR of hit times
    off_std=('beat_offset', np.std),          # std deviation of hit times
    vel_min=('velocity', "min"),              # quietest hit volume
    vel_max=('velocity', "max"),              # loudest hit volume
    vel_median =('velocity', np.median),      # median vol of all hits
    vel_mean=('velocity', np.mean),           # mean vol of all hits
    vel_iqr=('velocity', iqr),                # IQR of hit volumes
    vel_std=('velocity', np.std)              # std deviation of hit vols
    )
    
  # smooth to 2 decimal places
  stats_1 = stats_1.round(2)
  
  # label instruments with text instead of instrument number
  instrument_list = stats_1.index.unique(level=1)._data
  instrument_map = dict(zip(instrument_list, mt.getInstruments(instrument_list)))
  
  # flip so headers are the beat number index 
  stats_2 = stats_1.T.stack(1).fillna(0)
  
  # Flesh out missing metric placeholders. This helps when there are
  # some metric positions that don't have any beats, it'll insert
  # columns that are missing, filling with zeros. 
  
  # Flesh out df with any missing quantize buckets
  fill_cols = pd.DataFrame(columns=[x for x in range(1, 17)])
  full_col_list = list(set().union(stats_2.columns, fill_cols.columns))
  stats_2 = stats_2.reindex(columns=full_col_list, fill_value=0)
  
  ## produce statistics table for each insrument
  
  # gather list of instruments
  keys=stats_2.index.unique(level=1)._data

  results = {}
  
  for i in keys:
    print('>> Gathering stats for instrument: {}'.format(instrument_map[i]))
    next_stats = stats_2[stats_2.index.isin([i], level=1)].droplevel('note')
    results[i] = next_stats
  
  return results # TODO return structure containing stats for each instrument

  

def iqr(x):
  '''IQR (Inter Qurtile Range) function
  Used by the df.groupby.agg() function
  '''
  q75, q25 = np.percentile(x, [75 ,25],interpolation='midpoint')
  iqr = q75 - q25
  return iqr
  


def system_stats(midi_file_data):
  '''After doing all the work above, I discovered that
  'DataFrameGroupBy.describe' does something similar in
  a different format. Useful to compare my results to the
  output here 

  Note in the output below, IQR = 75%-25%. Median = 50%, 
  which helps match up the data with the headers I used 
  in the DIY stats above.
  
  NOTE..
  - assumes parameter 'midi_file_data' has been 
  created/ formatted by loading a MIDI file
  using function 'data_prep.load_file()'
  '''
  stats_df = midi_file_data.drop(columns=['bar_beat_number', 'note', 'track_msg_num', 'total_ticks', 'beat_center', 'bar_number' ]).copy()
  
  stats_tmp = stats_df.groupby(by=['bar_beat_number', 'note']).describe()
  
  return stats_tmp


'''
Miscellaneous code below here
'''
  
def __now():
  ''' Utility function
  '''
  return datetime.datetime.now()

def test_function_call(some_param):
  print('Test function in {} called and worked! when: {},  param:{}'.format(__name__, __now(), some_param))

# debug log that module loaded
print('>> LOADING custom module, when: {}, module name: {}'.format(__now(), __name__))

if __name__ == '__main__':
  print('>> confirming {} module ran :) '.format(__name__))





