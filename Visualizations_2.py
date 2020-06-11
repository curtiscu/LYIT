"""
    
Visualtisation library code for dissertation: v1

Also see: https://github.com/curtiscu/LYIT/blob/master/Visualizations_2.ipynb

"""
import data_prep as dp
from data_prep import MidiTools as mt

class Visualizations:

  def __init__(self, plot_instruments=[36, 38, 51, 42]):
    self.file_name = None # string name of file
    self.file_df = None  # raw df data from file
    self.instr_in_file = None # list of instruments in file
    self.plot_instruments = plot_instruments # filter instruments to plot
    self.plot_from_measure=1  # start plotting from bar#
    self.plot_measures=2 # plot for number of bars
    self.instr_time_filtered_df = None # data filtered for plotting
    self.bins_per_bar = None  # quantize/ smallest beat level
    self.ts_num = None # time sig beats per bar
    self.bars_in_file = 0

  def __str__(self):
    return 'Viz_2 for file: {}'.format(self.file_name)

  def set_instruments(self, plot_instruments):
    ''' 
      Can be used to set the instruments that are plotted, and
      the order they're rendered in.
    '''
    self.plot_instruments = plot_instruments

  def load(self, file_name):
    self.file_name = file_name    
    self.file_df, f, mtt = dp.load_file(file_name)  # load data
    self.bins_per_bar = int(mtt.bins_per_bar()) # store quantize level for plotting
    self.ts_num = int(f.ts_num()) # store musical beats per bar, for plotting
    self.bars_in_file = mtt.bars_in_file()


  def configure_plot(self, plot_from_measure=1, plot_measures=2, plot_instruments=[36, 38, 51, 42]):
    '''
      MUST be called before a plot call, sets up environment
    '''
    self.plot_from_measure = plot_from_measure
    self.plot_measures = plot_measures
    self.plot_instruments = plot_instruments

    # selects bars to print
    _fr = plot_from_measure
    _to = plot_from_measure+plot_measures-1
    #print('  configure - set bar range from: {}, to: {}'.format(_fr, _to))
    time_filtered_df = self.file_df.loc[_fr:_to]

    # check/ log available instruments in file
    instr_in_file = time_filtered_df.note.unique()
    instr_in_file.sort()
    #print('  configure - instruments available to plot: {}, names: {}'.format(instr_in_file, mt.getInstruments(instr_in_file)))

    # filter instruments if need to
    if self.plot_instruments is None:
      self.plot_instruments = instr_in_file
    self.instr_time_filtered_df = time_filtered_df[time_filtered_df['note'].isin(self.plot_instruments)]

    # grab & store some other data for doing plots
    num_instruments = len(self.plot_instruments)
    names = mt.getInstruments(self.plot_instruments)
    #print('  configure - instruments will plot: {}, names: {}'.format(self.plot_instruments, names ))


  def do_grid_plot(self):

    # handy sortcut
    df = self.instr_time_filtered_df

    # setup beat/ bar indicators
    #total_small_beats = list(range(1.5, (self.bins_per_bar * self.plot_measures) , 1))
    total_small_beats = list(range(1, self.bins_per_bar * self.plot_measures, 1))
    big_beat_ticks = total_small_beats[:: int(self.bins_per_bar/ self.ts_num)]
    bar_ticks = big_beat_ticks[::self.ts_num]

    # debug
    #print('total_small_beats: {}'.format(total_small_beats))
    #print('big_beat_ticks: {}'.format( big_beat_ticks))
    #print('bar_ticks: {}'.format( bar_ticks))

    # Build data structure required by broken_barh
    bag_of_instruments = {}
    for i in self.plot_instruments:
      instrument_hits = df.loc[df['note'] == i]
      plot_positions = ((instrument_hits['bar_number'] - self.plot_from_measure) * self.bins_per_bar) + instrument_hits['bar_beat_number']
      #print('FOUND instrument: {}, plot_positions: {}'.format(i, plot_positions.values))
      instrument_hit_duples = []
      for next_hit in plot_positions.values:
        instrument_hit_duples.append((next_hit, 0.15))
        
      bag_of_instruments[i] = instrument_hit_duples

    #print('bag_of_instruments: {}'.format(bag_of_instruments))

    # object that provides colours for charts
    cycol = cycle('bgrcmykw')  

    # create list for x-axis markers in plot
    my_xticks = list(range(0, (self.bins_per_bar * self.plot_measures)+1, 1))

    #### SHOW PLOT
    
    # grab & store some other data for doing plots
    num_instruments = len(self.plot_instruments)
    names = mt.getInstruments(self.plot_instruments)

    fig, ax = plt.subplots()
    fig.set_size_inches(12*self.plot_measures, 1+num_instruments, forward=True)
    
    # loop for each instrument
    y_axis = 0
    for i in self.plot_instruments:
      y_axis += 10
      ax.broken_barh(bag_of_instruments[int(i)], (y_axis, 7), facecolors=next(cycol));


    ax.set_ylim(5, (num_instruments*10)+10);
    ax.set_xlim(0, len(total_small_beats)+2);
    ax.set_xlabel('Plot for bars {} - {}'.format(self.plot_from_measure, self.plot_from_measure+ self.plot_measures-1));
    ax.set_yticks(list(range(15, ((num_instruments+1)*10)+5, 10)));
    ax.set_yticklabels(names);

    #Add lines for beats and bars
    for next_beat_marker in big_beat_ticks:
      ax.axvline(next_beat_marker, color='black', linestyle='dotted', linewidth=2); 
    for next_bar_marker in bar_ticks:
      ax.axvline(next_bar_marker, color='black', linestyle='solid', linewidth=2); 

    ax.set_xticks(my_xticks);
    ax.grid(True);
    
      #Add horizontal and vertical lines
    plt.axhline(0, color='black', linestyle='dotted', linewidth=2);  #horizontal line

    plt.show();



  def do_offset_plot(self):
    df = self.instr_time_filtered_df      # handy sortcut

    # setup beat/ bar indicators
    #total_small_beats = list(range(1.5, (self.bins_per_bar * self.plot_measures) , 1))
    total_small_beats = list(range(1, self.bins_per_bar * self.plot_measures, 1))
    big_beat_ticks = total_small_beats[:: int(self.bins_per_bar/ self.ts_num)]
    bar_ticks = big_beat_ticks[::self.ts_num]

    # Building data structures for plot
    bag_of_instrument_hits = {}
    bag_of_instrument_offsets = {}

    for i in self.plot_instruments:

      # filter to hits for next instrument
      instrument_hits = df.loc[df['note'] == i]

      # calculate grid positions of beats
      plot_positions = ((instrument_hits['bar_number'] - self.plot_from_measure) * self.bins_per_bar) + instrument_hits['bar_beat_number']
      #print('FOUND instrument: {}, plot_positions: {}'.format(i, plot_positions.values))
      instrument_hit_array = []
      for next_hit in plot_positions.values:
        instrument_hit_array.append(next_hit)
      bag_of_instrument_hits[i] = instrument_hit_array

      # calculate timing offsets for beats
      instrument_offset_array = []
      for next_offset in instrument_hits['beat_offset']:
        instrument_offset_array.append(next_offset)
      bag_of_instrument_offsets[i] = instrument_offset_array

    # create list for x-axis markers in plot
    my_xticks = list(range(0, (self.bins_per_bar * self.plot_measures)+1, 1))
    
    # debug
    #print('bag_of_instrument_hits: {}'.format(bag_of_instrument_hits))
    #print('bag_of_instrument_offsets: {}'.format(bag_of_instrument_offsets))
    #print('my_xticks: {}'.format(my_xticks))


    #### SHOW PLOT

    # grab & store some other data for doing plots
    num_instruments = len(self.plot_instruments)
    names = mt.getInstruments(self.plot_instruments)

    fig, ax = plt.subplots()

    fig.set_size_inches(12*self.plot_measures, 1+num_instruments, forward=True)
    

    # loop for each instrument
    for i in self.plot_instruments:
      ax.plot(bag_of_instrument_hits[i], bag_of_instrument_offsets[i], '-o', ms= 10, label=mt.mapInstrument(i));

    ax.set(xlabel='Plot for bars {} - {}'.format(self.plot_from_measure, self.plot_from_measure+ self.plot_measures-1), 
           ylabel='Beat offset from norm (ticks)', 
           title='timing offset data');

    # add chart lines
    ax.axhline(0, color='black', linestyle='dotted', linewidth=2);  #horizontal line

    for next_beat_marker in big_beat_ticks:
      ax.axvline(next_beat_marker, color='black', linestyle='dotted', linewidth=2); 
    for next_bar_marker in bar_ticks:
      ax.axvline(next_bar_marker, color='black', linestyle='solid', linewidth=2); 

    ax.set_xticks(my_xticks);
    ax.grid();
    ax.legend();

    plt.draw();


  def do_layer_plot(self, plot_type='off', window_size=2, instruments=[51]):
    '''
      Parameters..
        type:         'off' = timing offset plot, 'vel' = velocity plot
        window_size:  width of visible plot, in number of bars
        instruments:  array of MIDI note values, indiciting which
                      should be plotted.
    '''

    plot_types = {'off':'beat_offset', 'vel':'velocity'}
    metric_to_plot = plot_types[plot_type]

    plot_measure_size = window_size  # layer everything in single bar
    instr_to_plot = instruments  # just pick out the ride for now

    # grab & store some other data for doing plots
    num_instruments = len(instr_to_plot)
    names = mt.getInstruments(instr_to_plot)

    print('Layer plot, instrument: {}, bars in file: {}'.format(names, self.bars_in_file))

    fig, ax = plt.subplots()
    fig.set_size_inches(20*plot_measure_size, 8, forward=True)

    # setup beat/ bar indicators
    total_small_beats = list(range(1, self.bins_per_bar * plot_measure_size, 1))
    big_beat_ticks = total_small_beats[:: int(self.bins_per_bar/ self.ts_num)]
    bar_ticks = big_beat_ticks[::self.ts_num]

    plot_range = range(1, self.bars_in_file, plot_measure_size)

    for next_bar in [*plot_range]:
      # print('next bar: {}'.format(next_bar))

      self.configure_plot(plot_from_measure=next_bar, plot_measures=plot_measure_size, plot_instruments=instr_to_plot)
      df = self.instr_time_filtered_df      # handy sortcut

      # Building data structures for plot
      bag_of_instrument_hits = {}
      bag_of_instrument_metrics = {}  # values from either 'beat_offset', or 'velocity'

      for i in self.plot_instruments:

        # filter to hits for next instrument
        instrument_hits = df.loc[df['note'] == i]

        # calculate grid positions of beats
        plot_positions = ((instrument_hits['bar_number'] - self.plot_from_measure) * self.bins_per_bar) + instrument_hits['bar_beat_number']
        #print('FOUND instrument: {}, plot_positions: {}'.format(i, plot_positions.values))
        instrument_hit_array = []
        for next_hit in plot_positions.values:
          instrument_hit_array.append(next_hit)
        bag_of_instrument_hits[i] = instrument_hit_array

        # calculate timing offsets for beats
        instrument_metric_array = []
        for next_metric in instrument_hits[metric_to_plot]:
          instrument_metric_array.append(next_metric)
        bag_of_instrument_metrics[i] = instrument_metric_array

      # create list for x-axis markers in plot
      my_xticks = list(range(0, (self.bins_per_bar * plot_measure_size)+1, 1))
      
      # loop to plot each instrument
      for i in self.plot_instruments:
        ax.plot(bag_of_instrument_hits[i], bag_of_instrument_metrics[i], '-o', ms= 10, label=next_bar);

    #### SHOW PLOT

    ax.set(xlabel='Plot for instruments:{},  bars {}-{}'.format(names, 1, self.bars_in_file), 
          ylabel='{}'.format(metric_to_plot), 
          title='{} data'.format(metric_to_plot));

    # add chart lines
    ax.axhline(0, color='black', linestyle='dotted', linewidth=2);  #horizontal line

    for next_beat_marker in big_beat_ticks:
      ax.axvline(next_beat_marker, color='black', linestyle='dotted', linewidth=2); 
    for next_bar_marker in bar_ticks:
      ax.axvline(next_bar_marker, color='black', linestyle='solid', linewidth=2); 

    ax.set_xticks(my_xticks);
    ax.grid();
    ax.legend();

    plt.draw();
