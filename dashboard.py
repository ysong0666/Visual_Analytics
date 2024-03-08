import matplotlib.cm as cm
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import json
import os
import altair as alt
import pydeck as pdk
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objs as go
import matplotlib.colors
from re import S
from tkinter import N
import plotly.graph_objects as go
import math
from streamlit_plotly_events import plotly_events
import re
import time

# Setting styles for app
st.set_page_config(layout='wide',
                   page_title="VA System",
                   page_icon="⭐️")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
            min-width: 0px;
            max-width: 250px;
            }
    .main > div {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    footer {visibility: hidden;}
    .stRadio {
        padding-left: 2rem;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Hypothesis Driven Visual Analysis System")

# --------------------READ FILE----------------------- #
# paths for the files:
CSV_OG = "data/add_col.csv"
CSV_CAR = "data/car.csv"
BMP = "data/Lekagul_Roadways.bmp"
CARS = "data/cars.csv"
SENSORS = "data/sensor_location.csv"
PATH = "data/path_coordinate.csv"
PATHS = "data/all_path.csv"
# -------------------SET UP MAP----------------------- #
# Open BMP file
img = plt.imread(BMP)
img_arr = np.array(img)
back_img = Image.open(BMP)

# --------------------SET UP DF----------------------- #
# load csv into a dataframe
df = pd.read_csv(CSV_OG, index_col=0)
car = pd.read_csv(CSV_CAR)
sensors = pd.read_csv(SENSORS)
cars = pd.read_csv(CARS)
paths = pd.read_csv(PATH)
all_paths = pd.read_csv(PATHS)

# adding columns, data tranformation
trespassed = df[df['trespassed'] == True]
no_exit = car[car['left-park'] == False]

# adding gate_group column
df['gate_group'] = df['gate-name'].str.slice(stop=-1)
df.loc[df['gate_group'] == "ranger-bas", 'gate_group'] = "ranger-base"

# adding timeframe yr_mon column
df['yr_mon'] = df['Timestamp'].str.slice(stop=7)

# adding season column
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['month'] = df['Timestamp'].dt.month

paths["index"] = paths.index
paths['gate1_group'] = paths['gate1'].str.slice(stop=-1)
paths['gate2_group'] = paths['gate2'].str.slice(stop=-1)
paths.loc[paths['gate1_group'] == "ranger-bas", 'gate1_group'] = "ranger-base"
paths.loc[paths['gate2_group'] == "ranger-bas", 'gate2_group'] = "ranger-base"


def assign_season(mon):
    if mon in [12, 1, 2]:
        return 'Winter'
    elif mon in [3, 4, 5]:
        return 'Spring'
    elif mon in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


df['season'] = df['month'].apply(assign_season)

# group filter
# gate_groups = df['gate_group'].unique()
# selected_gate_group = st.selectbox('Select a Gate Group', options=gate_groups)
# filtered_counts = counts[counts['gate_group'] == selected_gate_group]
# fig.update_traces(x=filtered_counts['gate_group'], y=filtered_counts['Count'])

# car type filter

# season filter

# Setting up variables
car_types = df['car-type'].unique()
gate_groups = df['gate_group'].unique()
seasons = df['season'].unique()
car_ids = df['car-id'].unique()
selected_car_types = car_types
selected_groups = gate_groups
selected_seasons = seasons
selected_cars = [""]
gates = sensors["gate-name"].unique()
durations = cars["days"].unique()
temp = pd.to_datetime(df["Timestamp"], utc=True)
days = temp.dt.date.unique()

start_time = min(days)
end_time = max(days)

min_days = 0
max_days = max(durations)
pos_days = [i for i in range(min_days, max_days + 1)]
start_duration = min_days
end_duration = max_days

p_options = ["Valid Path", "Skipped Sensor"]
speed = all_paths["mph"].unique()
min_speed = 0
max_speed = math.ceil(max(speed))
speeds = [i for i in range(min_speed, max_speed + 1)]


# --------------------FUNCTIONS------------------------ #
# filters based on general range selection (apply to all graphs)
def filter():  # TODO: change to have more filters
    filtered = df[df['car-type'].isin(selected_car_types) & df['gate_group'].isin(selected_groups) & df['season'].isin(
        selected_seasons)]
    filtered = filtered[filtered['Timestamp'] >= str(start_time)]
    filtered = filtered[filtered['Timestamp'] <= str(end_time)]
    if selected_cars != [""]:
        filtered = filtered[filtered['car-id'].isin(selected_cars)]
    filtered = filtered[filtered['car-id'].isin(get_valid_cars())]
    return filtered


# getting cars with desired duration
def get_valid_cars():
    filtered_cars = cars[cars["days"] >= start_duration]
    filtered_cars = filtered_cars[filtered_cars["days"] <= end_duration]
    return filtered_cars["car-id"].unique()


# loading sensor traffic bar graph
def load_bar1(filtered):
    filtered_counts = filtered.groupby(
        ['yr_mon', 'car-type']).size().reset_index(name='Count')
    color_map = {'1': '#00429d', '2': '#4976b5', '2P': '#7dbcd7', '3': '#b1e1fc',
                 '4': '#ffa59e', '5': '#dd4c65', '6': '#93003a'}

    fig = px.bar(filtered_counts, x='yr_mon', y='Count',
                 color='car-type', barmode='stack', color_discrete_map=color_map)
    # fig.update_traces(x=filtered_counts['gate_group'], y=filtered_counts['Count'])
    fig.update_layout(autosize=True, width=500, height=250,
                      margin=dict(l=30, r=0, t=30, b=0), title="Traffic by Month")
    fig.update_xaxes(title='')
    # wrap the plotly chart in a container and set the justify-content CSS property to center
    with st.container():
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},
                        style={'justify-content': 'center', 'display': 'flex'})


def load_bar2(filtered):
    filtered_counts = filtered.groupby(
        ['gate_group', 'car-type']).size().reset_index(name='Count')
    color_map = {'1': '#00429d', '2': '#4976b5', '2P': '#7dbcd7', '3': '#b1e1fc',
                 '4': '#ffa59e', '5': '#dd4c65', '6': '#93003a'}
    fig = px.bar(filtered_counts, x='gate_group', y='Count',
                 color='car-type', barmode='stack', color_discrete_map=color_map)
    # fig.update_traces(x=filtered_counts['gate_group'], y=filtered_counts['Count'])
    fig.update_layout(autosize=True, width=500, height=250,
                      margin=dict(l=30, r=0, t=30, b=10), title="Traffic by Gate Group")
    fig.update_xaxes(title='')
    with st.container():
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},
                        style={'justify-content': 'center', 'display': 'flex'})


def filter_path():
    col1, col2, col3 = st.columns((1, 0.1, 1))
    with col1:
        path_option = st.selectbox(
            "Which kind of path do you want to analyze?", p_options)
    with col2:
        ""
    filtered_path = all_paths
    if path_option == "Skipped Sensor":
        filtered_path = filtered_path[filtered_path['path-id'] > 161]
    else:
        filtered_path = filtered_path[all_paths['path-id'] <= 161]
        with col3:
            start_speed, end_speed = st.select_slider(
                'Select speed(mph)', speeds, value=[min_speed, max_speed])
        filtered_path = filtered_path[filtered_path['mph'] >= start_speed]
        filtered_path = filtered_path[filtered_path['mph'] <= end_speed]
    if selected_cars != [""]:
        filtered_path = filtered_path[filtered_path['car-id'].isin(
            selected_cars)]

    filtered_p_c = paths[paths['gate1_group'].isin(selected_groups)]
    filtered_p_c = filtered_p_c[filtered_p_c['gate2_group'].isin(
        selected_groups)]
    valid_paths = filtered_p_c["index"].unique()

    filtered_path = filtered_path[filtered_path['path-id'].isin(valid_paths)]
    filtered_path = filtered_path[filtered_path['car-type'].isin(
        selected_car_types)]
    filtered_path = filtered_path[filtered_path['start-time']
                                  >= str(start_time)]
    filtered_path = filtered_path[filtered_path['end-time'] <= str(end_time)]
    valid_cars = get_valid_cars()
    filtered_path = filtered_path[filtered_path['car-id'].isin(valid_cars)]
    return filtered_path


def is_colored(a, b, c, diff):
    if abs(int(a) - int(b)) > diff:
        return True
    elif abs(int(a) - int(c)) > diff:
        return True
    elif abs(int(b) - int(c)) > diff:
        return True
    return False


def load_heatmap(filtered):
    s = []
    for g in gates:
        s.append(len(filtered[filtered['gate-name'] == g]))

    counted_sensors = sensors.copy()
    counted_sensors["count"] = s
    counted_sensors["index"] = counted_sensors.index

    # change black to white, white to black
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            pixel = img_arr[i, j]
            if np.array_equal(pixel, [255, 255, 255, 255]):
                img_arr[i, j] = [0, 0, 0, 255]
            elif is_colored(pixel[0], pixel[1], pixel[2], 20):
                img_arr[i, j] = [0, 0, 0, 255]
            else:
                img_arr[i, j] = [255, 255, 255, 255]

    # Create scatter plot
    scatter = go.Scatter(
        x=sensors['X'],
        y=sensors['Y'],
        mode='markers',
        marker=dict(size=18, color=s, colorscale='cividis',
                    opacity=0.9, colorbar=dict(thickness=15, title='Traffic', outlinewidth=0, len=0.7,)),
        text=sensors['gate-name'],
        hovertemplate='Gate Name: %{text}<br>X: %{x}<br>Y: %{y}<br>Traffic: %{marker.color}',
        hovertext=sensors['gate-name'],

    )
    layout = go.Layout(
        legend=dict(title='Traffic Levels', bgcolor='lightgray'),
        title='Park Map with Sensor Location Traffic',
        autosize=True,
        width=500,
        height=500,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest"
    )
    # Create image trace
    image = go.Image(
        z=img_arr,
        hoverinfo='none')

    # Create figure with both traces
    fig = go.Figure(data=[image, scatter],
                    layout=layout)
    fig.update_xaxes(showticklabels=False, showgrid=False, tickcolor="white")
    fig.update_yaxes(showticklabels=False, showgrid=False, tickcolor="white")

    # st.plotly_chart(fig)
    fig.update_layout(
        autosize=True,
        width=480,
        height=480,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0))

    # display chart using plotly_events
    with st.container():
        selected_points = plotly_events(fig, click_event=True)

        if selected_points:
            if selected_points[0]["curveNumber"] == 1:
                idx = fig.data[1].hovertext[selected_points[0]['pointNumber']]
                dt1 = filtered[filtered["gate-name"] == idx]
                dt1 = dt1.iloc[:, 1:]
                if "displayed_df1" not in st.session_state or not st.session_state["displayed_df1"].equals(dt1):
                    st.session_state["displayed_timestamp1"] = time.time()
                    st.session_state["displayed_df1"] = dt1.copy()
            else:
                if "displayed_timestamp1" not in st.session_state or not st.session_state["displayed_df1"].empty:
                    st.session_state["displayed_timestamp1"] = time.time()
                    st.session_state["displayed_df1"] = pd.DataFrame()
        selected_points = ""


def make_blue(value, count):
    alpha = (value-min(count))/(max(count)-min(count))
    invalpha = 1 - alpha
    color = cm.get_cmap('cividis')(1-invalpha)  # invert the color map
    return 'rgb(' + str(int(color[0]*255)) + "," + str(int(color[1]*255)) + "," + str(int(color[2]*255)) + ")"


def load_path(filtered):
    s = []
    for g in gates:
        s.append(len(filtered[filtered['gate-name'] == g]))

    counted_sensors = sensors.copy()
    counted_sensors["count"] = s
    counted_sensors["index"] = counted_sensors.index

    filtered_path = filter_path()
    count = []
    for i in range(len(paths)):
        count.append(len(filtered_path[filtered_path['path-id'] == i]))

    counted_paths = paths.copy()
    counted_paths["count"] = count

    trace3_list = []
    middle_node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.Marker(
            opacity=0
        ),
        customdata=[]
    )

    for idx, row in counted_paths.iterrows():
        if row["count"] == 0:
            continue
        trace3 = go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(color=make_blue(row["count"], count), width=3),
            # marker=dict(
            #     color='Black',
            #     size=10),
            hoverinfo='none'
        )
        x0 = row["gate1x"]
        y0 = row["gate1y"]
        x1 = row["gate2x"]
        y1 = row["gate2y"]
        trace3['x'] = trace3['x'] + (x0, x1, None)
        trace3['y'] = trace3['y'] + (y0, y1, None)
        trace3_list.append(trace3)

        middle_node_trace['x'] += (((x0 + x1) / 2),)
        middle_node_trace['y'] += (((y0 + y1) / 2),)
        middle_node_trace['text'] += (
            "gate1: " + row["gate1"] + "<br>gate2: " + row["gate2"] + "<br>count: " + str(row["count"]),)
        middle_node_trace['customdata'] += (idx,)

    colorbar_trace = go.Scatter(x=[None],
                                y=[None],
                                mode='markers',
                                marker=dict(
                                    colorscale='cividis',
                                    showscale=True,
                                    cmin=-5,
                                    cmax=5,
                                    colorbar=dict(thickness=15, tickvals=[-5, 5],
                                                  ticktext=[
                                                      str(min(count)), str(max(count))],
                                                  outlinewidth=0, len=0.8,
                                                  )
    ),
        hoverinfo='none'
    )

    scatter_trace = go.Scatter(x=counted_sensors["X"],
                               y=counted_sensors["Y"],
                               mode='markers',
                               marker=dict(
        color='Black',
        size=10),
        text=counted_sensors["gate-name"],
        hoverinfo='text',
        customdata=counted_sensors["index"]
    )

    fig2 = go.Figure(layout={
        'xaxis': {'visible': False, 'showticklabels': False},
        'yaxis': {'visible': False, 'showticklabels': False},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    # for t in [*trace3_list, middle_node_trace]:
    fig2.add_traces([*trace3_list, middle_node_trace,
                    colorbar_trace, scatter_trace])
    fig2.update_layout(
        autosize=False,
        width=450,
        height=400,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0))
    fig2['layout']['yaxis']['autorange'] = "reversed"
    fig2.update_yaxes(showgrid=False, showticklabels=False)
    fig2.update_xaxes(showgrid=False, showticklabels=False)

    # this line displays the plot already
    selected_points2 = plotly_events(fig2, click_event=True)

    if selected_points2:
        if selected_points2[0]['curveNumber'] == len(fig2.data) - 3:
            idx = fig2.data[selected_points2[0]['curveNumber']
                            ].customdata[selected_points2[0]['pointNumber']]
            dt2 = filtered_path[filtered_path["path-id"] == idx]
            dt2 = dt2.iloc[:, 1:]
            if "displayed_df2" not in st.session_state or not st.session_state["displayed_df2"].equals(dt2):
                st.session_state["displayed_df2"] = dt2.copy()
                st.session_state["displayed_timestamp2"] = time.time()
        else:
            if "displayed_df2" not in st.session_state or not st.session_state["displayed_df2"].empty:
                st.session_state["displayed_timestamp2"] = time.time()
                st.session_state["displayed_df2"] = pd.DataFrame()
    selected_points2 = ""

# ranger/visitor traffic


def preprocess_data(df, time_period):
    # df.set_index('Timestamp', inplace=True)
    df = df.set_index('Timestamp')

    # Resample data based on the selected time period
    if time_period == 'Hourly':
        resampled_df = df.resample('H')
    elif time_period == 'Daily':
        resampled_df = df.resample('D')
    else:  # Monthly
        resampled_df = df.resample('M')

    # Aggregate the data
    aggregated_df = resampled_df['car-type'].value_counts().unstack().fillna(0)

    # Summarize the data for park (2P) and visitor(non-2P) cars
    aggregated_df['Ranger'] = aggregated_df['2P']
    aggregated_df['Visitor'] = aggregated_df.drop(columns=['2P']).sum(axis=1)
    return aggregated_df[['Ranger', 'Visitor']]


def plot_traffic(df, width=800, height=600):
    fig = px.line(df, x=df.index, y=['Visitor', 'Ranger'], labels={
                  'value': 'Number of Cars', 'variable': 'Car Type', 'Timestamp': 'Time'}, color_discrete_sequence=['#3d69af', '#f4777f'], width=width, height=height)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_layout(
        legend=dict(
            x=1,
            y=1.1,
            xanchor='right',
            yanchor='top',
            orientation='h'
        )
    )
    return fig

# loading sensor heatmap


def load_line(filtered):
    # Create columns for plot and dataframe
    plot_column, dataframe_column = st.columns((1, 1))

    with plot_column:
        time_period = st.radio(
            "", ("Daily", "Hourly", "Monthly"), horizontal=True)
        aggregated_df = preprocess_data(filtered, time_period)
        fig = plot_traffic(aggregated_df, width=900, height=350)
        fig.update_layout(margin=dict(l=20, r=20, t=0, b=10))
        with st.container():
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},
                            style={'justify-content': 'center', 'display': 'flex'})

    with dataframe_column:
        if "displayed_timestamp1" in st.session_state or "displayed_timestamp2" in st.session_state:
            if "displayed_timestamp1" not in st.session_state:
                if st.session_state["displayed_df2"].empty:
                    st.dataframe(filter(), use_container_width=True)
                else:
                    st.dataframe(
                        st.session_state["displayed_df2"], use_container_width=True)
            elif "displayed_timestamp2" not in st.session_state or st.session_state["displayed_timestamp1"] > st.session_state["displayed_timestamp2"]:
                if st.session_state["displayed_df1"].empty:
                    st.dataframe(filter(), use_container_width=True)
                else:
                    st.dataframe(
                        st.session_state["displayed_df1"], use_container_width=True)
            else:
                if st.session_state["displayed_df2"].empty:
                    st.dataframe(filter(), use_container_width=True)
                else:
                    st.dataframe(
                        st.session_state["displayed_df2"], use_container_width=True)
        else:
            def highlight_true(s):
                if s:
                    return 'background-color: coral'
                else:
                    return ''
            filtered = filtered[:3000]
            filtered = filtered.style.background_gradient(subset=['mph'], cmap='Blues')\
                .applymap(highlight_true, subset=['trespassed'])
            st.dataframe(filter(), use_container_width=True)


def get_view_no():
    if 'view_no' not in st.session_state:
        st.session_state.view_no = 1
    return st.session_state.view_no


def get_save_count():
    if 'save_count' not in st.session_state:
        st.session_state.save_count = 1
    return st.session_state.save_count


def increment_view_no():
    st.session_state.view_no += 1


def increment_save_count():
    st.session_state.save_count += 1


def run():
    view_no = get_view_no()
    save_count = get_save_count()
    col1, col2, col3 = st.columns((1, 1, 1))
    with col1:
        if st.button('Save View', key='Save_{}'.format(view_no)):
            # Define a unique name for the saved view
            saved_views[str(view_no)] = {
                'car_type': selected_car_types,
                'gate_group': selected_groups,
                'season': selected_seasons,
                'start_time': str(start_time),
                'end_time': str(end_time),
                'start_duration': start_duration,
                'end_duration': end_duration
            }
            with open(SAVED_VIEWS_FILE, 'w') as f:
                json.dump(saved_views, f)
            increment_view_no()
        increment_save_count()
        load_heatmap(filter())
    with col2:
        load_path(filter())
    with col3:
        load_bar1(filter())
        load_bar2(filter())

    load_line(df)


# --------------------SAVE VIEW FILE------------------------ #
# TODO: add save view button here
# Save the filter settings and corresponding graph to a file
# Define the name of the saved views file
SAVED_VIEWS_FILE = 'saved_views.json'
# Initialize the saved views dictionary
saved_views = {}
# Load saved views from file if it exists
if os.path.exists(SAVED_VIEWS_FILE):
    with open(SAVED_VIEWS_FILE, 'r') as f:
        saved_views = json.load(f)

# --------------------SIDE BAR------------------------ #
st.sidebar.subheader('Quick  Explore')
if st.sidebar.checkbox("Show Columns"):
    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)
if st.sidebar.checkbox("Trespassed"):
    def highlight_true(s):
        if s:
            return 'background-color: coral'
        else:
            return ''
    trespassed = trespassed.style.background_gradient(subset=['mph'], cmap='YlOrRd')\
        .applymap(highlight_true, subset=['trespassed'])
    st.dataframe(trespassed)

if st.sidebar.checkbox("Never left park"):
    st.write(no_exit)
if st.sidebar.checkbox("Daily Traffic"):
    col1, col2 = st.columns(2)
    color_map = {'entrance': '#00429d', 'camping': '#4976b5', 'gate': '#7dbcd7', 'general-gate': '#b1e1fc',
                 'ranger-bas': '#ffa59e', 'ranger-stop': '#dd4c65', '6': '#93003a'}
    with col1:
        df['hour'] = df['Timestamp'].dt.hour
        hourly_counts = df.groupby(
            ['hour', 'gate_group']).size().reset_index(name='count')
        fig = px.bar(hourly_counts, x='hour', y='count', color='gate_group', barmode='stack',
                     color_discrete_map=color_map, hover_name="gate_group", title="Traffic by Hour and Gate Group")
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        day_of_week_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                           3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['day_of_week'] = df['day_of_week'].map(day_of_week_map)
        daily_counts = df.groupby(
            ['day_of_week', 'gate_group']).size().reset_index(name='count')
        fig = px.bar(daily_counts, x='day_of_week', y='count', color='gate_group', barmode='stack',
                     color_discrete_map=color_map, hover_name="gate_group", title="Traffic by Day of the Week and Gate Group")
        st.plotly_chart(fig, use_container_width=True)


st.sidebar.subheader('Advanced Explore')
if st.sidebar.checkbox("Select filters"):
    selected_cars_string = st.sidebar.text_input('Input car-id')
    selected_cars = re.split(', ', selected_cars_string)
    selected_groups = st.sidebar.multiselect(
        'select gate groups', options=gate_groups, default=gate_groups)
    selected_car_types = st.sidebar.multiselect(
        'Select car types', options=car_types, default=car_types)
    selected_seasons = st.sidebar.multiselect(
        'Select seasons', options=seasons, default=seasons)
    start_time, end_time = st.sidebar.select_slider(
        'Select timeframe', days, value=[min(days), max(days)])
    start_duration, end_duration = st.sidebar.select_slider(
        'Select duration', pos_days, value=[min_days, max_days])
# Display a list of saved views to the user
if st.sidebar.checkbox('Show saved views'):
    st.subheader('Saved Views')
    for view_name, view_data in saved_views.items():
        st.write(
            f'{view_name}: car types: {view_data["car_type"]}, gate groups: {view_data["gate_group"]}, seasons: {view_data["season"]}, start time: {view_data["start_time"]}, '
            f'end time: {view_data["end_time"]}, start_duration: {view_data["start_duration"]}, end_duration: {view_data["end_duration"]}')
        if st.button(f'Load View {view_name}', key='Load_{}'.format(view_name)):
            selected_car_types = view_data["car_type"]
            selected_groups = view_data["gate_group"]
            selected_seasons = view_data["season"]
            start_time = view_data["start_time"]
            end_time = view_data["end_time"]
            start_duration = view_data["start_duration"]
            end_duration = view_data["end_duration"]
            run()
            # Code to load filter settings and corresponding graph data from saved_views[view_name]

# create a text area for user input
# if st.sidebar.checkbox('Add note'):
note = st.sidebar.text_area("Enter your note here")
# save the note when the user clicks the save button
# if st.sidebar.button("Save"):
# insert code to save the note to a file or database here
# st.sidebar.success("Note saved!")

# ------- make 3 columns to evenly distribute the plots
run()
