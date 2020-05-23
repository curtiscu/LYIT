
"""

Library code for dissertation: v2, 2020-05-21

"""



# imports
import pandas as pd
import mido 
from mido import MidiFile
import math



class MIDI_File_Wrapper:
  '''
  Utility wrapper for loading, parsding a mido.MidiFile object
  '''

  # column headers for internal data frame 
  # containing MIDI messages loaded from file
  vel_col = 'velocity'
  note_col = 'note'
  type_col = 'msg_type'
  time_col = 'delta_ticks'
  cum_ticks_col = 'total_ticks'
  raw_col = 'raw_data'
  cum_ms_col = 'total_seconds'

  # used for setting order of columns in data model df
  __column_in_order = [type_col, time_col, cum_ticks_col, cum_ms_col, note_col, vel_col, raw_col]

  def __init__(self, file_name, note_map = None):
    self.my_file_name = file_name   # string filename
    self.my_file_midi = None        # mido.MidiFile instance   
    self.my_tempo = None            # stored as mido.Message instance
    self.my_time_sig = None         # stored as mido.Message instance
    self.df_midi_data = None        # DataFrame holding MIDI messages
    self.instruments = None         # list of instruments played in file
    self.note_map = note_map        # changes event notes on load
    self.last_note_on = 0           # stores last performed event in file

    # load file and gather data...
    self.parse_file()


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

    # load file and gather data...
    self.parse_file()
  
  def parse_file(self):
    '''
    File must be: MIDI type 0 only;  one and only one tempo and time_sig meta messages in file. 
    '''

    print('FILE: {}'.format(self.my_file_name))

    # load file
    midi_file = MidiFile(self.my_file_name)
    self.my_file_midi = midi_file 

    # make sure it's MIDI type 0 (single track) ...
    if midi_file.type != 0:
      raise ValueError('ERROR! Can only process type 0 files, this file is type: {}'.format(midi_file.type))

    print('    tracks: {}'.format(midi_file.tracks))

    # another check for single track ...
    if len(midi_file.tracks) != 1:
      raise ValueError('ERROR! Need a single MIDI track, this file has: {}, {}'.format(midi_file.tracks, midi_file))

    # parse messages for time_sig and tempo info ..
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
    self.__load_df()

    print('    last note_on: {}'.format(self.last_note_on))

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


  def __load_df(self):
    df_setup = []
    
    # build df structure from the MIDI file...
    for msg in self.my_file_midi.tracks[0]:
      df_setup.append(
          {
              self.type_col: msg.dict()['type'],
              self.time_col: msg.dict()['time'],
              self.note_col: None if 'note' not in msg.dict() else msg.dict()['note'],
              self.vel_col: None if 'velocity' not in msg.dict() else msg.dict()['velocity'],
              self.raw_col:  str(msg.dict()) # saves whole message in case needed later
          } 
      )

    df_tmp = pd.DataFrame(df_setup)

    # tweak data types, change from 'object' columns to 'string'  ...
    df_tmp[self.type_col] = df_tmp[self.type_col].astype('string')
    df_tmp[self.raw_col] = df_tmp[self.raw_col].astype('string')
    
    # add cumulative tick count column, used to store a running total
    # giving time a message appears in the performance/ MIDI file.
    df_tmp[self.cum_ticks_col] = df_tmp[self.time_col].cumsum()

    # remember the tick position of last note_on in file
    self.last_note_on = df_tmp[df_tmp[self.type_col] == 'note_on'].tail(1)[self.cum_ticks_col].values[0]

    # add cumulative milliseconds from start of file
    # NOTE: this timing needs to be recalculated if the tempo
    #         is ever changed!!!
    df_tmp[self.cum_ms_col] = df_tmp.apply(self.__row_to_seconds, axis=1)

    # apply note mappings, store in new column
    if self.note_map != None:
      df_tmp[self.note_col] = df_tmp[self.note_col].map(self.note_map, na_action='ignore')

    # grab list of instruments used in file
    drum_stuff = df_tmp.note.unique()
    drum_stuff.sort()
    self.instruments = drum_stuff[pd.notnull(drum_stuff)]  # filters NaN 

    

    # set column order
    df_tmp = df_tmp[self.__column_in_order]
  
    # store final df
    self.df_midi_data = df_tmp



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

  # bucket size for quantizing, set to 1/16 notes
  def bin_size(self):
    return int(self.ticks_per_16()) 

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



class GrooveMidiTools:
  '''
  tools specifically for working with this dataset
  https://magenta.tensorflow.org/datasets/
  '''

  # mappings taken from https://magenta.tensorflow.org/datasets/groove#drum-mapping
  mappings = {22: 42,	# Closed Hi-Hat
              26: 46, 	# Open Hi-Hat
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

 
def test_function_call(name):
  print('test function called worked! :)  {}'.format(name))


# debug log that module loaded
print('LOADING - data_prep.py module name is: {}'.format(__name__))


if __name__ == '__main__':
    print('yay, curtis module ran :) ')
	
