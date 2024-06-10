import streamlit as st
import numpy as np
import time
import io
from PIL import Image
import matplotlib.pyplot as plt
import glob
from enum import Enum
from io import StringIO

class Commands(Enum):
    COMMENT = 0
    MOVE = 1
    OTHER = 2
    TOOLCHANGE = 3

from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
import re

# glength = {"G0":0,"G1":0,"G2":0,"G3":0,"Time":0}

@dataclass
class GcodeLine:
    command: Tuple[str, int]
    params: Dict[str, float]
    comment: str

    def __post_init__(self):
        if self.command[0] == 'G' and self.command[1] in (0, 1, 2, 3):
            self.type = Commands.MOVE
        elif self.command[0] == ';':
            self.type = Commands.COMMENT
        elif self.command[0] == 'T':
            self.type = Commands.TOOLCHANGE
        else:
            self.type = Commands.OTHER

    @property
    def command_str(self):
        return f"{self.command[0]}{self.command[1] if self.command[1] is not None else ''}"

    def get_param(self, param: str, return_type=None, default=None):
        """
        Returns the value of the param if it exists, otherwise it will the default value.
        If `return_type` is set, the return value will be type cast.
        """
        try:
            if return_type is None:
                return self.params[param]
            else:
                return return_type(self.params[param])
        except KeyError:
            return default

    def update_param(self, param: str, value: Union[int, float]):
        if self.get_param(param) is None:
            return
        if type(value) not in (int, float):
            raise TypeError(f"Type {type(value)} is not a valid parameter type")
        self.params[param] = value
        return self.get_param(param)

    def delete_param(self, param: str):
        if self.get_param(param) is None:
            return
        self.params.pop(param)

    @property
    def gcode_str(self):
        command = self.command_str

        def param_value(param):
            value = self.get_param(param)
            is_flag_parameter = value is True
            if is_flag_parameter:
                return ""
            return value

        params = " ".join(f"{param}{param_value(param)}" for param in self.params.keys())
        comment = f"; {self.comment}" if self.comment != '' else ""
        if command == ';':
            return comment
        return f"{command} {params} {comment}".strip()

class GcodeParser:
    def __init__(self, gcode: str, include_comments=False):
        self.gcode = gcode
        self.lines: List[GcodeLine] = get_lines(self.gcode, include_comments)
        self.include_comments = include_comments

def get_lines(gcode, include_comments=False):
    # regex = r'(?!; *.+)(G|M|T|g|m|t)(\d+)(([ \t]*(?!G|M|g|m)\w(".*"|([-+\d\.]*)))*)[ \t]*(;[ \t]*(.*))?|;[ \t]*(.+)'
    # regex = r'(?!; *.+)(G|M|T|X|Y|Z|g|m|t|x|y|z)(\d+)(([ \t]*(?!G|M|g|m)\w(".*"|([-+\d\.]*)))*)[ \t]*(;[ \t]*(.*))?|;[ \t]*(.+)'
    regex = r'(?!; *.+)(G|M|T|X|Y|Z|g|m|t|x|y|z)([-+]?\.?\d+\.?\d*)(([ \t]*(?!G|M|g|m)\w(".*"|([-+\d\.]*)))*)[ \t]*(;[ \t]*(.*))?|;[ \t]*(.+)'
    # white spaces and comments start with ';' and in '()'
    clean_pattern = re.compile(r'N\d+')
    clean_pattern1 = re.compile(r'\(.*?\)')
    lines_clean = re.sub(clean_pattern, '', gcode)
    lines_clean = re.sub(clean_pattern1, '', lines_clean)
    regex_lines = re.findall(regex, lines_clean)
    lines = []
    for line in regex_lines:
        if line[0]:
          # print(line[0],line[1],line[2])
          if line[0].upper()== 'X' or line[0].upper()== 'Y' or line[0].upper()== 'Z':
            command = ('G',1)
            params = split_params(''.join(line[:3]))
          else :
            command = (line[0].upper(), int(line[1]))
            comment = line[-2]
            params = split_params(line[2])

        elif include_comments:
            command = (';', None)
            comment = line[-1]
            params = {}

        else:
            continue

        lines.append(
            GcodeLine(
                command=command,
                params=params,
                comment=comment.strip(),
            ))

    return lines

def element_type(element: str):
    if re.search(r'"', element):
        return str
    if re.search(r'\..*\.', element):
        return str
    if re.search(r'[+-]?\d*\.', element):
        return float
    if re.search(r'[+-]?\d*\.\d+', element):
        return float
    return int

def split_params(line):
    regex = r'((?!\d)\w+?)(".*"|(\d+\.?)+|[-+]?\d*\.?\d*)'
    elements = re.findall(regex, line)
    params = {}
    for element in elements:
        if element[1] == '':
            params[element[0].upper()] = True
            continue
        params[element[0].upper()] = element_type(element[1])(element[1])

    return params

def arcG2R(x0,x1,y0,y1,z0,z1,r):
  import numpy as np
  X, Y, Z = [], [], []
  zdiff = z1-z0
  for d in range(0,100,10):
    rcos = r*np.cos(d*np.pi/180)
    rsin = r*np.sin(d*np.pi/180)
    z = z0 + d*zdiff/90
    if x1>x0 and y1<y0 :
      X.append(x0+rsin)
      Y.append(y1+rcos)
    elif x1<x0 and y1<y0:
      X.append(x1+rcos)
      Y.append(y0-rsin)
    elif x1<x0 and y1>y0:
      X.append(x0-rsin)
      Y.append(y1-rcos)
    elif x1>x0 and y1>y0:
      X.append(x1-rcos)
      Y.append(y0+rsin)
    Z.append(z)

  return [X,Y,Z]

def arcG3R(x0,x1,y0,y1,z0,z1,r):
  import numpy as np
  l = 0
  X, Y, Z = [], [], []
  zdiff = z1-z0
  for d in range(0,100,10):
    rcos = r*np.cos(d*np.pi/180)
    rsin = r*np.sin(d*np.pi/180)
    z = z0 + d*zdiff/90
    if x1<x0 and y1>y0 :
      X.append(x1+rcos)
      Y.append(y0+rsin)
    elif x1<x0 and y1<y0:
      X.append(x0-rsin)
      Y.append(y1+rcos)
    elif x1>x0 and y1<y0:
      X.append(x1-rcos)
      Y.append(y0-rsin)
    elif x1>x0 and y1>y0:
      X.append(x0+rsin)
      Y.append(y1-rcos)
    Z.append(z)

  return [X,Y,Z]

def plot2d_live(df,time_feed=10):
  import matplotlib.pyplot as plt
  import numpy as np
  from time import sleep
  fig = plt.figure(figsize=(12, 8))
  glength = {"G0":0,"G1":0,"G2":0,"G3":0,"Time":0}
  # syntax for 3-D projection
  # ax = fig.add_subplot(projection='3d')
  ax = plt.axes()
  colors = ['b-','r-','g-','c-','m-','y-','k-']*100 # [Blue,Red,Green,Cyan,Magenta,Yellow,Black,White(w)]
  Xmax, Ymax = max(np.array(df['X'])), max(np.array(df['Y']))
  Xmin, Ymin = min(np.array(df['X'])), min(np.array(df['Y']))
  Xlen, Ylen = Xmax-Xmin, Ymax-Ymin
  ax.set_xlim(Xmin-0.1*Xlen,Xmax+0.1*Xlen)
  ax.set_ylim(Ymin-0.1*Ylen,Ymax+0.1*Ylen)
  ax.set_aspect('equal') #auto
  for i in range(len(df)):
    t = df.loc[i,'T']
    f = df.loc[i,'F']
    G = df.loc[i,'GCode']
    i_1 = 0 if i==0 else i-1
    x0,x1 = int(df.loc[i_1,'X']),int(df.loc[i,'X'])
    y0,y1 = int(df.loc[i_1,'Y']),int(df.loc[i,'Y'])
    z0,z1 = int(df.loc[i_1,'Z']),int(df.loc[i,'Z'])
    if G == 'G2':
      I,J = df.loc[i,'I'],df.loc[i,'J']
      R = df.loc[i,'R']
      if R :
        r = R
      else :
        r = np.sqrt(I**2+J**2)
      X,Y,Z = arcG2R(x0,x1,y0,y1,z0,z1,r)
      ax.plot(X, Y, colors[t-1],linewidth=lineWt)
      for i in range(len(X)-1):
         glength['G2'] += np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
      glength['Time'] += glength['G2']/f

    elif G == 'G3':
      I,J = df.loc[i,'I'],df.loc[i,'J']
      R = df.loc[i,'R']
      if R :
        r = R
      else :
        r = np.sqrt(I**2+J**2)
      X,Y,Z = arcG3R(x0,x1,y0,y1,z0,z1,r)
      ax.plot(X, Y, colors[t-1],linewidth=lineWt)
      for i in range(len(X)-1):
         glength['G3'] += np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
      glength['Time'] += glength['G3']/f

    else :
      if G == "G0":
         color = 'c-.'
         glength['G0'] += np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
         glength['Time'] += glength['G0']/f
      else :
         color = colors[t-1]
      ax.plot([x0,x1], [y0,y1], color,linewidth=lineWt)
      glength['G1'] += np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
      glength['Time'] += glength['G1']/f
    the_plot.pyplot(plt)
    sleep(time_feed/f)
  st.session_state['G0'] = glength['G0']
  st.session_state['G1'] = glength['G1']
  st.session_state['G2'] = glength['G2']
  st.session_state['G3'] = glength['G3']
  st.session_state['GT'] = glength['Time']
  
def plot3d_live(df,time_feed=10):
  import matplotlib.pyplot as plt
  from mpl_toolkits import mplot3d
  import numpy as np
  from time import sleep
  fig = plt.figure(figsize=(12, 8))
  glength = {"G0":0,"G1":0,"G2":0,"G3":0,"Time":0}
  
  # syntax for 3-D projection
  # ax = fig.add_subplot(projection='3d')
  ax = plt.axes(projection ='3d')
  colors = ['m-','b-','g-','c-','r-','y-','k-']*100 # [Blue,Red,Green,Cyan,Magenta,Yellow,Black,White(w)]
  # time_feed = 10   # mm

  Xmax, Ymax, Zmax = max(np.array(df['X'])), max(np.array(df['Y'])), max(np.array(df['Z']))
  Xmin, Ymin, Zmin = min(np.array(df['X'])), min(np.array(df['Y'])), min(np.array(df['Z']))
  Xlen, Ylen, Zlen = Xmax-Xmin, Ymax-Ymin, Zmax-Zmin
  ax.set_xlim3d(Xmin-0.1*Xlen,Xmax+0.1*Xlen)
  ax.set_ylim3d(Ymin-0.1*Ylen,Ymax+0.1*Ylen)
  ax.set_zlim3d(Zmin-0.1*Zlen,Zmax+0.1*Zlen)
  ax.set_aspect('equal') #auto
  for i in range(len(df)):
    t = df.loc[i,'T']
    f = df.loc[i,'F']
    G = df.loc[i,'GCode']
    i_1 = 0 if i==0 else i-1
    x0,x1 = int(df.loc[i_1,'X']),int(df.loc[i,'X'])
    y0,y1 = int(df.loc[i_1,'Y']),int(df.loc[i,'Y'])
    z0,z1 = int(df.loc[i_1,'Z']),int(df.loc[i,'Z'])
    if G == 'G2':
      I,J = df.loc[i,'I'],df.loc[i,'J']
      R = df.loc[i,'R']
      if R :
        r = R
      else :
        r = np.sqrt(I**2+J**2)
      X,Y,Z = arcG2R(x0,x1,y0,y1,z0,z1,r)
      ax.plot3D(X, Y, Z, colors[t-1],linewidth=lineWt)
      for i in range(len(X)-1):
         glength['G2'] += np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
      glength['Time'] += glength['G2']/f

    elif G == 'G3':
      I,J = df.loc[i,'I'],df.loc[i,'J']
      R = df.loc[i,'R']
      if R :
        r = R
      else :
        r = np.sqrt(I**2+J**2)
      X,Y,Z = arcG3R(x0,x1,y0,y1,z0,z1,r)
      ax.plot3D(X, Y, Z, colors[t-1],linewidth=lineWt)
      for i in range(len(X)-1):
         glength['G3'] += np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
      glength['Time'] += glength['G3']/f

    else :
      if G == "G0":
         color = 'c-.'
         glength['G0'] += np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
         glength['Time'] += glength['G0']/f
      else :
         color = colors[t-1]
      ax.plot3D([x0,x1], [y0,y1], [z0,z1], color,linewidth=lineWt)
      glength['G1'] += np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
      glength['Time'] += glength['G1']/f

    the_plot.pyplot(plt)
    sleep(time_feed/f)
  st.session_state['G0'] = glength['G0']
  st.session_state['G1'] = glength['G1']
  st.session_state['G2'] = glength['G2']
  st.session_state['G3'] = glength['G3']
  st.session_state['GT'] = glength['Time']

def readGcode(gcode):
  import pandas as pd
  dict_gcode = {}
  param_list = []
  # with open(gpath+'gcode/DR0F01_C.tap', 'r') as f:
  #   gcode = f.read()
  # parsed_gcode = GcodeParser(gcode)

  parsed_gcode = GcodeParser(gcode)
  lines = parsed_gcode.lines
  x, y, z, feed, t = 0., 0., 0., 100, 1
  column = ["GCode","X","Y","Z","I","J","R","F","T"]
  for i in range(len(lines)):
    if lines[i].get_param('F') is not None :
      feed = lines[i].get_param('F')
    if lines[i].get_param('T') is not None :
      t = lines[i].get_param('T')
    if lines[i].command[0] == 'T':
      t = lines[i].command[1]
    if lines[i].type == Commands.MOVE :
      if lines[i].get_param('X') is not None :
        x = lines[i].get_param('X')
      if lines[i].get_param('Y') is not None :
        y = lines[i].get_param('Y')
      if lines[i].get_param('Z') is not None :
        z = lines[i].get_param('Z')
      if lines[i].get_param('R') is not None :
        r = lines[i].get_param('R')
      else :
        r = ''
      if lines[i].get_param('I') is not None :
        ii = lines[i].get_param('I')
      else :
        ii = ''
      if lines[i].get_param('J') is not None :
        jj = lines[i].get_param('J')
      else :
        jj = ''
      param_list.append([lines[i].command[0]+str(lines[i].command[1]),x,y,z,ii,jj,r,feed,t])
  df = pd.DataFrame(param_list,columns=column)
  return df

# Initialization
if 'G0' not in st.session_state:
    st.session_state['G0'] = 0
if 'G1' not in st.session_state:
    st.session_state['G1'] = 0
if 'G2' not in st.session_state:
    st.session_state['G2'] = 0
if 'G3' not in st.session_state:
    st.session_state['G3'] = 0
if 'GT' not in st.session_state:
    st.session_state['GT'] = 0

stringio = ''  
# sidebar title
st.sidebar.write('Enter Gcode to analyse')

# image source selection
list_select = ['Use a sample Gcode', 'Use your own Gcode']
option = st.sidebar.selectbox('Select upload method', list_select)
valid_datasets = glob.glob('data/*.txt')

fname = ''
if option == list_select[0]:
    # st.sidebar.write('Select a sample dataset')
    fname = st.sidebar.selectbox('Select from existing list',
                                 valid_datasets)
    if fname :
      with open(fname ,'r',encoding='utf-8') as f:
        stringio = f.read()

elif option == list_select[1]:
    # st.sidebar.write('Select an Gcode to upload')
    file_upload = st.sidebar.file_uploader('Choose a gcode file to upload',
                                     accept_multiple_files=False)
    if file_upload is not None:
      strio = StringIO(file_upload.getvalue().decode("utf-8"))
      stringio = strio.read()
    # Can be used wherever a "file-like" object is accepted:
# To convert to a string based IO:

# st.write(stringio)
texts = st.text_area(label='Gcode',value=stringio,height=400)

df_plot = readGcode(texts)

st.button('Tool path 2D',on_click=lambda : plot2d_live(df_plot))
st.button('Tool path 3D',on_click=lambda : plot3d_live(df_plot))
# if st.button('Tool path 3D') :
#    gl = plot3d_live(df_plot)
lineWt = st.number_input('Line width',0.25,5.0,0.7,0.1) 
the_plot = st.pyplot(plt)
st.text(f"G0 Length : {st.session_state['G0']:.2f} mm")
st.text(f"G1 Length : {st.session_state['G1']:.2f} mm")
st.text(f"G2 Length : {st.session_state['G2']:.2f} mm")
st.text(f"G3 Length : {st.session_state['G3']:.2f} mm")
st.text(f"Total Time : {st.session_state['GT']:.2f} sec")
# Delete all the items in Session state
for key in st.session_state.keys():
    del st.session_state[key]
# st.text_area(value=f'G0 length : {gl["G0"]} mm\nG1 length : {gl["G1"]} mm\nG2 length : {gl["G2"]} mm\nG3 length : {gl["G3"]} mm\nTotal time : {gl["Time"]} sec')
# def get_img(i):
#     # Create a bytes buffer to save the plot
#     line.set_ydata(data[i:max_x+i])
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     # Open the PNG image from the buffer and convert it to a NumPy array
#     image = np.array(Image.open(buf))
#     # Close the buffer
#     buf.close()
#     return image

# def get_frame():
#     return np.random.randint(0, 255, size=(10,10,3))

# my_image = st.image(get_frame(), caption='Random image', width=600)
# my_image = st.image(get_img(0), caption='Random image', width=600)

# for i in range(100):
#     time.sleep(0.1)
#     my_image.image(get_img(i), caption='Random image', width=600)