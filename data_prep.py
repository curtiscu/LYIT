
"""

Prep library code for dissertation: v3, 2020-06-02
Loads MIDI files using MIDO library and does
a bunch of preprocessing of the data

"""



# imports
import pandas as pd
import mido 
from mido import MidiFile
import math
import datetime
from collections import namedtuple
import stats_n_features as sf

class MIDI_File_Wrapper:
  '''
  Utility wrapper for loading, parsding a mido.MidiFile object
  '''

  # column headers for internal data frame 
  # containing MIDI messages loaded from file
  track_msg_number_col = 'track_msg_num'
  vel_col = 'velocity'
  note_col = 'note'
  type_col = 'msg_type'
  time_col = 'delta_ticks'
  cum_ticks_col = 'total_ticks'
  raw_col = 'raw_data'
  cum_ms_col = 'total_seconds'
  channel_col = 'channel'

  # used for setting order of columns in data model df
  __column_in_order = [track_msg_number_col, type_col, time_col, cum_ticks_col, cum_ms_col, note_col, vel_col, raw_col]

  def __init__(self, file_name, note_map = None, note_filter = None):
    self.my_file_name = file_name   # string filename
    self.my_file_midi = None        # mido.MidiFile instance   
    self.my_tempo = None            # stored as mido.Message instance
    self.my_time_sig = None         # stored as mido.Message instance
    self.df_midi_data = None        # DataFrame holding MIDI messages
    self.instruments = None         # list of instruments played in file
    self.note_map = note_map        # changes event notes on load
    self.note_filter = note_filter  # filters instruments/ note-on on file loading
    self.last_note_on = 0           # stores last performed event in file
    self.first_note_on = 0          # stores first performed event in file

    # load file and gather data...
    self.__parse_file()


  # For call to str(). Prints readable form 
  def __str__(self): 
    return str('file: {}'.format(self.my_file_midi))

  def reset_content(self):
    '''
    Reloads MIDI file and recreates object, EXCEPT, it doesn't 
    reset the filters set during object creation
    '''
    self.my_file_midi = None        # mido.MidiFile instance   
    self.my_tempo = None            # stored as mido.Message instance
    self.my_time_sig = None         # stored as mido.Message instance
    self.df_midi_data = None        # DataFrame holding MIDI messages
    self.instruments = None         # list of instruments played in file
    self.last_note_on = 0           # stores last performed event in file
    self.first_note_on = 0          # stores first performed event in file

    # load file and gather data...
    self.__parse_file()
  
  def __parse_file(self):
    '''
      Loads some required metadata from the file then calls
      __load_messages() to handle parsing individual messages
      into primary dataframe
    '''

    global log_dict
    
    print('FILE name: {}'.format(self.my_file_name))

    # load file
    midi_file = MidiFile(self.my_file_name)
    self.my_file_midi = midi_file 
    print('    loaded file: {}'.format(midi_file))

    # parse messages for time_sig and tempo info, searches
    # across all tracks, rather than per track ..
    for msg in midi_file:
      if msg.type == 'time_signature':
        print('    time sig: {}'.format(msg))

        # make sure no time sig changes
        if self.my_time_sig != None:
          raise ValueError('ERROR! more than one time sig: {}, {}'.format(self.my_time_sig, msg))
        self.my_time_sig = msg

      elif msg.type == 'set_tempo':
        print('    tempo: {}'.format(msg))

        # make sure no tempo changes
        if self.my_tempo != None:
          raise ValueError('ERROR! more than one tempo: {}, {}'.format(self.my_tempo, msg))
        self.my_tempo = msg

    # now check we actually have tempo and time_sig set, or complain...
    if self.my_time_sig is None:
      raise ValueError('ERROR! no time signature found: {}'.format(midi_file))
    if self.my_tempo is None:
      raise ValueError('ERROR! no tempo found: {}'.format(midi_file))
      
    # load MIDI messages from file into DF
    self.__load_messages()

    # more debug
    print('    note_on span - first tick: {} , last tick: {} '.format(self.first_note_on, self.last_note_on))

    # quick debug to show instruments in file
    good, bad = MidiTools.getInstruments2(self.instruments)
    print('    good instruments: {}, {}'.format(len(good), good))
    if len(bad) > 0:
      print('    ____ ERR! INVALID PERCUSSION INSTRUMENTS: {}, {}'.format(len(bad),bad))
    

    
    
    

  def tempo_us(self):
    ''' Tempo in microseconds'''
    return self.my_tempo.tempo

  def tempo_bpm(self):
    ''' Tempo in bpm'''
    return mido.tempo2bpm(self.tempo_us())

  def ticks(self):
    ''' Returns number of MIDI ticks configured in this file'''
    return self.my_file_midi.ticks_per_beat

  def length(self):
    ''' returns running time in seconds'''
    return self.my_file_midi.length

  def msg_counts(self):
    ''' handy for debug '''
    return self.df_midi_data['msg_type'].value_counts()

  def ts_num(self):
    ''' Time signature numerator (top number)'''
    return self.my_time_sig.numerator

  def ts_denom(self):
    ''' Time signature denominator (bottom number) '''
    return self.my_time_sig.denominator

  def first_hit(self):
    return self.first_note_on

  def last_hit(self):
    return self.last_note_on

  def calculate_seconds(self, ticks_since_start):
    ''' 
    Takes elapsed ticks since start of files, returns 
        position in file in absolute seconds'''

    # uses ticks and tempo saved from file loading time..
    return mido.tick2second(ticks_since_start, self.ticks(), self.tempo_us())


  def __row_to_seconds(self, row):
    return self.calculate_seconds(row[self.cum_ticks_col])


  def __load_messages(self):  
    '''
      Careful about handling file type here ..
      - could be a problem if not MIDI type 0 (single track)
      - limited testing, maybe a problem if type 1 (multiple synchronous tracks)
      - most likely a problem if type 2 (multiple asynchronous tracks)
    '''

    global log_dict
    
    # shorthand
    _f = self.my_file_midi
    
    # DEBUG check for track count ...
    print('    track count: {}, tracks: {}'.format(len(_f.tracks), _f.tracks))
    
    # shouldn't be a problem if MIDI type 0 (single track)
    # maybe a problem if type 1 (multiple synchronous tracks)
    # most likely a problem if type 2 (multiple asynchronous tracks)
    print('    MIDI file type: {}'.format(_f.type))
    
    if _f.type == 0:
      pass # all good
    elif _f.type == 1: 
      print('    CAREFUL - THIS MAY BE A PROBLEM! file of type: {}'.format(_f.type))
    else:
      raise ValueError("ERROR! Unknown MIDI file type, unable to proceed with file type: {}, tracks: {}, midi_file: {}".format(_f.type, _f.tracks, _f))

    # logs
    log_dict['file_type'] = [_f.type]

    # Next loop will ..
    # - make dfs from all tracks
    # - tweak data types
    # - create ticks cumsum col

    track_dfs = []

    for track_number, next_track in enumerate(_f.tracks):
      print('    > processing track: {}'.format(next_track))
      
      df_setup = [] # capture message data for next df
        
      # IMPORTANT NOTE.. 
      # 2 ways can extract messages, from track object
      # or from file object. Extracting from tracks gives
      # time in ticks, the file object converts it to secs
      
      # build df structure from the next MIDI track...
      for msg_number, msg in enumerate(next_track):
        
        track_msg_number = '{}:{}'.format(track_number, msg_number)
        #print('      > processing msg: {}'.format(track_msg_number))
        #print('      > processing msg: {}'.format(msg))
        
        # append next MIDI message info
        df_setup.append(
          {
            self.track_msg_number_col: track_msg_number, # msg position in track
            self.type_col: msg.dict()['type'],
            self.time_col: msg.dict()['time'],
            self.channel_col: None if 'channel' not in msg.dict() else msg.dict()['channel'],
            self.note_col: None if 'note' not in msg.dict() else msg.dict()['note'],
            self.vel_col: None if 'velocity' not in msg.dict() else msg.dict()['velocity'],
            self.raw_col:  str(msg.dict()) # saves whole message in case needed later
          } 
        )

      next_df = pd.DataFrame(df_setup) # create the df for the current track

      # tweak data types, change from 'object' columns to 'string'  ...
      next_df[self.type_col] =next_df[self.type_col].astype('string')
      next_df[self.raw_col] =next_df[self.raw_col].astype('string')
      
      # add cumulative tick count column, used to store a running total
      # giving time a message appears in the performance/ MIDI file.
      next_df[self.cum_ticks_col] =next_df[self.time_col].cumsum()

      track_dfs.append(next_df)  # add message data captured in current df to the list
        
    # concat all track dfs, required for multi-track MIDI file
    df_tmp = pd.concat(track_dfs)

    # sort according to cumsum col, reindex/ new index
    df_tmp.sort_values([self.cum_ticks_col], ignore_index=True, inplace=True)

    # NOTE: at this point, all msgs in file are loaded and aggregated into 
    #       single df, and ordered according to MIDI tick position
    
    # remember the tick position of first note_on in file
    self.first_note_on = df_tmp[df_tmp[self.type_col] == 'note_on'].head(1)[self.cum_ticks_col].values[0]
    
    # remember the tick position of last note_on in file
    self.last_note_on = df_tmp[df_tmp[self.type_col] == 'note_on'].tail(1)[self.cum_ticks_col].values[0]

    # add cumulative milliseconds from start of file
    # NOTE: this timing needs to be recalculated if the tempo
    #         is ever changed!!!
    df_tmp[self.cum_ms_col] = df_tmp.apply(self.__row_to_seconds, axis=1)

    # NOTE: next section allows for filtering of instruments
    # from loaded MIDI file
    
    # show list of raw/ unfiltered instruments loaded from file...
    print('    __notes pre-filter: {}'.format(self.clean_list(df_tmp.note.unique())))
    
    # apply pre-filter to remove select instruments
    if self.note_filter != None:
      print('    __applying filter: {}'.format(self.note_filter))
      df_tmp = df_tmp[~df_tmp[self.note_col].isin(self.note_filter)]
      
    # show list instruments after filtering
    print('    __notes post filter: {}'.format(self.clean_list(df_tmp.note.unique())))
      
    # apply note mappings, store in new column
    # note: this is used primarily to map from Roland MIDI
    # to General MIDI drum notes, specified in 'simplified_mapping'
    if self.note_map != None:
      df_tmp[self.note_col] = df_tmp[self.note_col].map(self.note_map, na_action='ignore')

    # grab list of filtered/ mapped instruments
    self.instruments = self.clean_list(df_tmp.note.unique())
 
    # set column order
    df_tmp = df_tmp[self.__column_in_order]
  
    # store final df
    self.df_midi_data = df_tmp
    
  # utility function to tidy up instrument list
  # takes list/ array, sorts it, removes NaN, returns it
  def clean_list(self, some_list):
    some_list.sort()
    return some_list[pd.notnull(some_list)]  # filters NaN 




class MidiTimingTools:


  # all parameters required
  def __init__(self, label, file_ticks_per_beat, us_per_beat, time_sig_numerator, time_sig_denominator, last_note_on):
    self.label = label                                # pretty label, for handy reference
    self.file_ticks_per_beat = file_ticks_per_beat    # from MIDI file header
    self.time_sig_numerator = time_sig_numerator      #   "
    self.time_sig_denominator = time_sig_denominator  #   "
    self.us_per_beat = us_per_beat                    # from MIDI file meta message
    self.last_note_on = last_note_on                  # from MIDI file data


  # For call to str(). Prints readable form, tests all 
  # function calls to build debug string output. 
  def __str__(self): 
    return str("LABEL: {} \n  Ticks p/beat: {} \n  BPM: {} \n  time sig: {}/ {} \n  bars in file: {} \n  beats in file: {} \n  ticks in file: {} \n  bins: {} \n  beats: {}"
    .format(self.label, 
            self.file_ticks_per_beat,
            self.bpm(),
            self.time_sig_numerator,
            self.time_sig_denominator,
            self.bars_in_file(),
            self.beats_in_file(), 
            self.ticks_in_file(), 
            self.get_bins(), 
            self.get_beats()))

  def bpm(self):
    return (60 * 1000000) / self.us_per_beat

  # ts = time signature
  def ts_ticks_per_beat(self):
    return self.file_ticks_per_beat * ( 4/ self.time_sig_denominator )

  def ticks_per_bar(self):
    return self.ts_ticks_per_beat() * self.time_sig_numerator

  def ticks_per_8(self):
    return self.file_ticks_per_beat/ 2
    
  def ticks_per_16(self):
    return self.file_ticks_per_beat / 4

  # calculates total bars, round up for whole bars
  def bars_in_file(self):
    return math.ceil(self.last_note_on / self.ticks_per_bar()) 

  def ticks_in_file(self):
    return int(self.bars_in_file() * self.ticks_per_bar()) # total ticks to render (file_range)

  def beats_in_file(self):
    return self.bars_in_file() * self.time_sig_numerator

  # bucket size for quantizing, hardwired here to 1/16 notes
  # perhaps this might be best being configurable?
  def bin_size(self):
    return int(self.ticks_per_16()) 

  # how many bins in a bar, e.g. if quantizing by 16th
  # notes, this'll return 16, if by 8th notes, return 8.
  def bins_per_bar(self):
    return self.ticks_per_bar()/ self.bin_size()

  # takes a bar# to start, and how many bars from
  # there to count. handy if you need to get a start
  # and end number of ticks for a range of bars.
  def get_tick_range(self, start_bar, number_of_bars):
    start_tick = (self.ticks_per_bar() * (start_bar - 1)) - self.bin_size()/2
    end_tick = start_tick + (self.ticks_per_bar() * number_of_bars)
    return start_tick, end_tick

  def get_bins(self):   # my_bins
    my_bin_size = self.bin_size()
    file_range = self.ticks_in_file()
    return range(0 - (int(my_bin_size/ 2)), file_range + my_bin_size, my_bin_size)

  def get_beats(self):
    my_bin_size = self.bin_size()
    file_range = self.ticks_in_file()
    return range(0, file_range + my_bin_size, my_bin_size) 

  # NOTE - don't think I actually need this at all
  def calculated_bins(self, cumulative_ticks_series):
	  return pd.cut(cumulative_ticks_series, bins=self.get_bins(), right=False)

  # takes series with cumulative ticks since start of file for the 
  # MIDI note_on events, and returns a series stating the centre
  # of the beat for each given MIDI event
  def assigned_beat_location(self, cumulative_ticks_column):
    #return pd.cut(cumulative_ticks_column.values, bins=self.get_bins(), right=False, labels=self.get_beats())
    return pd.cut(cumulative_ticks_column, bins=self.get_bins(), right=False, labels=self.get_beats())


  def get_offsets(self, cumulative_ticks_column):
    my_beats = self.assigned_beat_location(cumulative_ticks_column)
    tmp_dict = dict(enumerate(my_beats.cat.categories))
    beat_centers = my_beats.cat.codes.map(tmp_dict)

    offsets = cumulative_ticks_column - beat_centers

    # this will return 'beat_centers' as Int
    #return beat_centers, offsets  

    # this will return 'my_beats' as Categorical
    return my_beats, offsets


class MidiTools:
  '''
  Convert to/ from MIDI notes to percussion instrumentz
  As per http://www.midi.org/techspecs/gm1sound.php
  '''

  note2Instrument = { 35: "Acoustic Bass Drum",
                36: "Bass Drum 1 (36)",
                37: "Side Stick (37)", 
                38: "Acoustic Snare (38)",
                39: "Hand Clap (39)",
                40: "Electric Snare (40)",
                41: "Low Floor Tom (41)",
                42: "Closed Hi Hat (42)",
                43: "High Floor Tom (43)",
                44: "Pedal Hi-Hat (44)",
                45: "Low Tom (45)",
                46: "Open Hi-Hat (46)",
                47: "Low-Mid Tom (47)",
                48: "Hi-Mid Tom (48)",
                49: "Crash Cymbal 1 (49)",
                50: "High Tom (50)",
                51: "Ride Cymbal 1 (51)",
                52: "Chinese Cymbal (52)",
                53: "Ride Bell (53)",
                54: "Tambourine (54)",
                55: "Splash Cymbal (55)",
                56: "Cowbell (56)",
                57: "Crash Cymbal 2 (57)",
                58: "Vibraslap (58)",
                59: "Ride Cymbal 2 (59)",
                60: "Hi Bongo (60)",
                61: "Low Bongo (61)",
                62: "Mute Hi Conga (62)",
                63: "Open Hi Conga (63)",
                64: "Low Conga (64)",
                65: "High Timbale (65)",
                66: "Low Timbale (66)",
                67: "High Agogo (67)",
                68: "Low Agogo (68)",
                69: "Cabasa (69)",
                70: "Maracas (70)",
                71: "Short Whistle (71)",
                72: "Long Whistle (72)",
                73: "Short Guiro (73)",
                74: "Long Guiro (74)",
                75: "Claves (75)",
                76: "Hi Wood Block (76)",
                77: "Low Wood Block (77)",
                78: "Mute Cuica (78)",
                79: "Open Cuica (79)",
                80: "Mute Triangle (80)",
                81: "Open Triangle (81)" }
  
  def mapInstrument(midi_note):
    '''
    Takes MIDI note number, returns None if not found, otherwise 
    returns a string name of the percussion instrument
    '''
    answer = None
    if midi_note in MidiTools.note2Instrument:
      answer = MidiTools.note2Instrument[midi_note]

    return answer

  def getInstruments(instrument_list):
    '''
    Takes a list of MIDI numeric notes, returns a list
    of string names of instruments played on this track
    '''
	
    result = []
	
    if instrument_list is not None:
      # NOTE: concise notation copied from https://stackoverflow.com/a/38702484
      result = [*map(MidiTools.mapInstrument, instrument_list)]
    
    return result

    
  def getInstruments2(instrument_list):

    good = {}
    bad = []

    for next in instrument_list:
      if next in MidiTools.note2Instrument:
        good[next] = MidiTools.note2Instrument[next]
      else:
        bad.append(next)

    return good, bad


# maps from Roland MIDI notes to General MIDI standard
# taken from https://magenta.tensorflow.org/datasets/groove#drum-mapping
simplified_mapping = {22: 42,	# Closed Hi-Hat
                      26: 46, # Open Hi-Hat
                      36: 36,	# Bass
                      37: 38,	# Snare 
                      38: 38,	# Snare 
                      40: 38,	# Snare 
                      42: 42,	# Closed Hi-Hat
                      43: 43,	# High Floor Tom
                      44: 42,	# Closed Hi-Hat
                      45: 47,	# Low-Mid Tom
                      46: 46,	# Open Hi-Hat
                      47: 47,	# Low-Mid Tom
                      48: 50,	# High Tom
                      49: 49,	# Crash Cymbal
                      50: 50,	# High Tom
                      51: 51,	# Ride Cymbal
                      52: 49,	# Crash Cymbal
                      53: 51,	# Ride Cymbal
                      55: 49,	# Crash Cymbal
                      57: 49,	# Crash Cymbal
                      58: 43,	# High Floor Tom
                      59: 51}	# Ride Cymbal
 
def __now():
  return datetime.datetime.now()
  
mt = MidiTools()

def load_file(file_name, filter_err_buckets=True, note_filter=[44]):
  '''
    Convenience function to collect steps for
    loading, initial preprocessing and feature
    generation from a MIDI file.
    
    file_name = String name of file to load
    note_filter = list/ array of instrument numbers to ignore/ filter
            default filter is '44', removes hi hat with foot.
    filter_err_buckets: If >1 instrument strike for same instrument 
      in a beat_position, our algorithm doesn't work that detailed so strip
      them all out.
  '''
  
  global log_dict
  
  midi_file = MIDI_File_Wrapper(file_name, simplified_mapping, note_filter)
  
  # logs
  log_dict['ts_num'] = [midi_file.ts_num()]
  log_dict['ts_den'] = [midi_file.ts_denom()]
  log_dict['tempo_us'] = [midi_file.tempo_us()]
  log_dict['tempo_bpm'] = [midi_file.tempo_bpm()]
  log_dict['PPQN'] = [midi_file.ticks()]
  log_dict['first_hit'] = [midi_file.first_hit()]
  log_dict['last_hit'] = [midi_file.last_hit()]
  
  # some shortcuts
  f = midi_file
  f_df = f.df_midi_data
  
  
  #### SETUP TIMING BINS (MAIN DF)

  # MTT object for parsing file and
  # calculating crticial time metrics
  mtt = MidiTimingTools(file_name, 
                        f.ticks(),  
                        f.tempo_us(), 
                        f.ts_num(), 
                        f.ts_denom(), 
                        f.last_hit())

  # values needed these for making MultiIndex later
  quantize_level = mtt.bins_per_bar()
  bars_in_file = mtt.bars_in_file()
  tp_beat = mtt.ts_ticks_per_beat()
  tp_bin = mtt.bin_size()
  
  # log
  log_dict['bars_in_file'] = [bars_in_file]
  log_dict['quantize_level'] = [quantize_level]
  log_dict['ticks_per_beat'] = [tp_beat]
  log_dict['ticks_per_bin'] = [tp_bin]
  log_dict['ms_per_bin'] = [mido.tick2second(tp_bin, f.ticks(), f.tempo_us())]

  # DEBUG
  print('    bar info - bars in file: {}, bar quantize level: {}'.format(bars_in_file, quantize_level))
  print('    tick info - ticks per time sig beat: {}, ticks per quantize bin: {}'.format(tp_beat, tp_bin))

  # capture timing data from MidiTimingTools in df...
  beats_col, offsets_col = mtt.get_offsets(f_df[f.cum_ticks_col])
  f_df['beat_offset'] = offsets_col
  f_df['beat_center'] = beats_col
  f_df['file_beat_number'] = pd.Categorical(f_df.beat_center).codes

  # uses ticks and tempo saved from file loading time..
  make_offset_ms_col = lambda x: mido.tick2second(x, f.ticks(), f.tempo_us())
  f_df['beat_offset_ms'] = f_df['beat_offset'].map(make_offset_ms_col)

  # make a copy, for now..
  tmp_df = f_df.copy(deep=True)

  # filter to only note_on events
  tmp_df = tmp_df[tmp_df['msg_type'] == 'note_on'].copy() 


  # sort out bar column
  tmp_df['bar_number'] = (tmp_df.file_beat_number // quantize_level) + 1
  # add column for beat within the bar index
  tmp_df['bar_beat_number'] = (tmp_df.file_beat_number % 16) + 1

  # sort out types
  tmp_df['bar_number'] = tmp_df['bar_number'].astype(int)
  tmp_df['bar_beat_number'] = tmp_df['bar_beat_number'].astype(int)
  tmp_df['velocity'] = tmp_df['velocity'].astype(int)
  tmp_df['note'] = tmp_df['note'].astype(int)
  tmp_df['track_msg_num'] = tmp_df['track_msg_num'].astype('string')


  # drop other columns we don't need
  tmp_df.drop(columns=[ 'msg_type', 
                        'delta_ticks', 
                        'total_seconds',  
                        'raw_data', 
                        'file_beat_number' ], inplace=True)


  #### SET REQUIRED INDEXES (MAIN DF)
  
  tmp_df.set_index(['bar_number', 'bar_beat_number', 'note'], inplace=True, drop=False)
  
  
  #### FILTER ERROR BUCKETS (MAIN DF)
  
  if filter_err_buckets:
    print('    > checking for errs...')
    err_buckets = get_error_buckets(tmp_df) # parse for problem beats
    # log
    log_dict['err_buckets'] = [err_buckets.shape]
    if err_buckets.size == 0:
      print('    ...no errors to see here')
    else: # handle buckets > 1 hit for instrument
      #display(err_buckets)
      print('    __ tmp_df before: {}'.format(tmp_df.shape))
      print('    __ err_buckets removed: {}'.format(err_buckets.shape))
      tmp_df = tmp_df.drop(err_buckets.index).copy() # remove errs
      print('    __ tmp_df after: {}'.format(tmp_df.shape))
      

  
  else:
    print('    > DISABLED - err bucket filtering.')



  #### GATHER OTHER BITS, BASED ON MAIN DF
  
  stats_df = sf.gather_stats(tmp_df) # parse to gather stats
  tight_df = get_tight_df(tmp_df)
    
  
  #### STORE CHANGES TO MAIN DF
  
  # replace MIDI_File_Wrapper dataframe with new one...
  f.df_midi_data = tmp_df
 
  
  ####  RETURN RESULTS
  
  # return all of the finished DataFrame, MIDI_File_Wrapper, 
  # and the MidiTimingTools, stats_df summary, and highly filtered
  # tight_df .. the caller can decide which of those to keep
  return f.df_midi_data, f, mtt, stats_df, tight_df
  

# returns a list of problematic beat locations
# that have >1 note strike for a given instrument
# in any single 16th note beat position
def get_error_buckets(midi_file_data):
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
  
'''
  Collection object to hold in a single place all information
  related to a performance loaded from a MIDI file.

  drummer_id = String
  file_df = complete DataFrame of data, unfiltered
  file_wrapper = MidiFileWrapper instance
  tools = MidiTimingTools instance
  stats_df = DataFrame fleshed out with additional features/ metrics
  tight_df = highly filtered to just the final kick, snare, hihats instruments
'''
PerformanceData = namedtuple('PerformanceData' , 'drummer_id style file_df file_wrapper tools stats_df tight_df tight_style_df')

meta_file = '/content/drive/My Drive/LYIT/Dissertation/data/eval_data.csv'

def load_meta_file():
  '''
    Loads the files specified in..
      '/content/drive/My Drive/LYIT/Dissertation/data/eval_data.csv'
  '''

  eval_df = pd.read_csv(meta_file, dtype = {"drummer": "string", 
                                            "session" : "string", 
                                            "id": "string", 
                                            "style": "string", 
                                            "time_signature" : "string",
                                            "midi_filename" : "string",
                                            "long_midi_filename" : "string",
                                            "split" : "string"})
  return eval_df



# INIT - call to reset at start 
log_dict = {}  # single files' logs 
all_logs_df = None # holds aggregated logs when load_all_data() is done. 

def get_last_logs():
  """
  
  
  """
  return all_logs_df

def load_all_data(filter_err_buckets=True):
  """
  Bulk loads MIDI data files. 
    
  filter_err_buckets: see load_file(filter_err_buckets). If False, 'file_df' has
    all performance notes as the MIDI messages were in the file, none dropped 
    or filtered.
    
  Filters to tunes in style..
    ['funk/groove1', 'soul/groove3', 'soul/groove4', 'hiphop/groove6', 'rock/groove8']
  """
  
  global all_logs_df
  global log_dict
  
  eval_df = load_meta_file()
  
  # NOTE: these are labels for each 'style' (1-10) the drummers
  # were asked to play, after reviewing, the following were deemed
  # most suitable/ usable as data for the project
  song_styles = ['funk/groove1', 'soul/groove3', 'soul/groove4', 'hiphop/groove6', 'rock/groove8']
  eval_df = eval_df[eval_df['style'].isin(song_styles)]
  
  # container holding all results, used to return
  # the final results at end of loading process
  all_drummer_data = {} 
  
  # tmp placeholder, collects file dfs to
  # be merged into combined master df
  songs_dfs = [] 
    
  log_dfs = [] # tmp container for all file logs

  # iterate over each file meta data
  for index, row in eval_df.iterrows():

    # reset logging for next loaded file 
    log_dict = {}
    
    # access data using column names
    next_drummer = row['drummer']
    long_name = row['long_midi_filename']
    short_name = row['midi_filename']
    style = row['style']

    # write some logging for this file
    log_dict['file_name'] = [short_name]
    log_dict['style'] = [style]
    log_dict['drummer'] = [next_drummer]
    
    
    # load all data for single file
    print('BULK LOAD: {}, {}, {}'.format(next_drummer, short_name, style))
    file_df, file_wrapper, mtt, stats_df, tight_df = load_file(long_name, filter_err_buckets)
    
    # append some additional meta info, add 'style' (i.e. song category) to full file_df
    file_df['style'] = style
    tight_style_df = tight_df.copy()
    tight_style_df['style'] = style

    # add tuple of data elements to dict with filename as key
    all_drummer_data[long_name] = PerformanceData(next_drummer, 
                                                  style, 
                                                  file_df, 
                                                  file_wrapper, 
                                                  mtt, 
                                                  stats_df, 
                                                  tight_df, 
                                                  tight_style_df)

    # master_df: used to make final single df of all
    # songs, all styles, all drummers
    df1 = tight_style_df.copy()
    df1.insert(0, 'drummer_ID', next_drummer) # add it as first col
    df1.reset_index(drop=True, inplace=True)
    songs_dfs.append(df1)
    
    # when done processing a file, append info to logs
    log_dfs.append(pd.DataFrame.from_dict(log_dict))
    

  # FINAL PROCESSING, LOGGING, ETC.
  
  # master_df: merge all into single df, drop indexes, reset default index
  all_songs_df = pd.concat(songs_dfs)
  all_songs_df.reset_index(inplace=True, drop=True)
  all_drummer_data['master_df'] = all_songs_df
    
  # when done loading all file, concat log dfs
  all_logs_df = pd.concat(log_dfs, ignore_index=True)
  
  return all_drummer_data


def get_tight_df(file_df):
  '''
    Takes the verbose dataframe and returns a slimmed down
    one, filtered to include notes from just kick, snare, hh 
    closed, hh open, and ride. This list is then further shrunk
    with 51 (ride), 42 (closed hh), and 46 (open hh) all merged 
    & analysed as a single right hand hh pattern. Notes are then 
    replaced with human readable english descriptions for convenience
    The end result is a df with just kick, snare, hats, the 3 core
    items used for analysis.
  '''
  
  # filter to core cols
  df1 = file_df.filter(items=['note',	'velocity',	'beat_offset', 'beat_offset_ms', 'bar_beat_number'], axis=1).copy()
  
  # drop index we don't need, it's still a column
  df1.index = df1.index.droplevel(level='note')

  # view info on instruments
  print('   > raw instruments: {}'.format(df1['note'].unique()))
  print('   {}'.format(MidiTools.getInstruments(df1['note'].unique())))
  print('   counts before final filter: {}'.format(df1.groupby('note')['note'].count())) # debug

  # filter notes to kick, snare, hh closed, hh open, and ride
  df1 = df1[df1.note.isin([36, 38, 51, 42, 46])]
  print('   counts after final filter: {}'.format(df1.groupby('note')['note'].count())) # debug

  # Replace note numbers with user friendly english descriptions
  # 51 (ride), 42 (closed hh), and 46 (open hh) all merged & analysed as
  # single right hand hh pattern with the end result always giving kick, 
  # snare, hats, the 3 core items to analyse
  df1.note.replace({36: 'kick', 38: 'snare', 51:'hh', 42:'hh', 46:'hh' }, inplace=True)
  print('    counts after final merge: {}'.format(df1.groupby('note')['note'].count())) # debug, view after final merging
  return df1
    

def test_function_call(some_param):
  print('Test function in data_prep.py called and worked! when: {},  param:{}'.format(__now(), some_param))
  


# debug log that module loaded
print('>> LOADING custom module, when: {}, module name: {}'.format(__now(), __name__))

if __name__ == '__main__':
  print('>> confirming data_prep.py module ran :) ')
	
