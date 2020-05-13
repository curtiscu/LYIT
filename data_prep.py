
"""

Library code for dissertation: v2, 2020-05-13


"""


# imports
import pandas as pd
import mido 
from mido import MidiFile

# debug
import datetime


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



class MidiTools:
  '''
  Convert to/ from MIDI notes to percussion instrumentz
  As per http://www.midi.org/techspecs/gm1sound.php
  '''

  note2Instrument = { 35: "Acoustic Bass Drum",
                36: "Bass Drum 1",
                37: "Side Stick", 
                38: "Acoustic Snare",
                39: "Hand Clap",
                40: "Electric Snare",
                41: "Low Floor Tom",
                42: "Closed Hi Hat",
                43: "High Floor Tom",
                44: "Pedal Hi-Hat",
                45: "Low Tom",
                46: "Open Hi-Hat",
                47: "Low-Mid Tom",
                48: "Hi-Mid Tom",
                49: "Crash Cymbal 1",
                50: "High Tom",
                51: "Ride Cymbal 1",
                52: "Chinese Cymbal",
                53: "Ride Bell",
                54: "Tambourine",
                55: "Splash Cymbal",
                56: "Cowbell",
                57: "Crash Cymbal 2",
                58: "Vibraslap",
                59: "Ride Cymbal 2",
                60: "Hi Bongo",
                61: "Low Bongo",
                62: "Mute Hi Conga",
                63: "Open Hi Conga",
                64: "Low Conga",
                65: "High Timbale",
                66: "Low Timbale",
                67: "High Agogo",
                68: "Low Agogo",
                69: "Cabasa",
                70: "Maracas",
                71: "Short Whistle",
                72: "Long Whistle",
                73: "Short Guiro",
                74: "Long Guiro",
                75: "Claves",
                76: "Hi Wood Block",
                77: "Low Wood Block",
                78: "Mute Cuica",
                79: "Open Cuica",
                80: "Mute Triangle",
                81: "Open Triangle" }
  
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
    # NOTE: concise notation copied from https://stackoverflow.com/a/38702484
    return [*map(MidiTools.mapInstrument, instrument_list)]
    
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
  mappings = {22: 42,
              26: 46,
              36: 36,
              37: 38,
              38: 38,
              40: 38,
              42: 42,
              43: 43,
              44: 42,
              45: 47,
              46: 46,
              47: 47,
              48: 50,
              49: 49,
              50: 50,
              51: 51,
              52: 49,
              53: 51,
              55: 49,
              57: 49,
              58: 43,
              59: 51}


def test_function_call(name):
  print('test function called worked! :)  {}'.format(name))


# debug log that module loaded
print('LOADING - data_prep.py module name is: {}'.format(__name__))


if __name__ == '__main__':
    print('yay, curtis module ran :) ')
	
