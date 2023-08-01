import matplotlib.pyplot as plt
import re
import csv
import numpy as np
import pandas as pd


def rnti_filter(rnti_path):
    input_data = open(rnti_path, 'r')
    rnti_list = []
    rnti_final = []
    for line in input_data:
        if "RNTI" in line:
            index1 = line.find("=")
            index2 = line.find(')')
            rnti_list.append(str(hex(int(line[index1+1:index2]))))
    for rnti in rnti_list:
        if rnti not in '0xfffe':
            rnti_final.append(rnti)
    return rnti_final


def filter_data(log_path, csv_write_down, csv_write_up, rntilist):
    input_data = open(log_path, 'r')
    for line in input_data:
        line = line.split(' ')

        rntilist = ['3f45', '4527', '1c0d', '33cd', '2d8b', '20d3', '2767', '3361',
                    '3ba5', '48fb', '2077', '2591', '2b8b', '304d', 'd39', '2c9b']

        for rnti in rntilist:
            if 'rnti=%s,' % rnti in line:
                if '[MAC' in line or '[HI]' in line:
                    break
                else:
                    tbs_string = [s for s in line if "tbs" in s]
                    rnti_string = [s for s in line if "rnti" in s]
                    uldl_string = [s for s in line if "DL" in s or 'UL' in s]
                    tbs_match = [line[0], rnti_string[0][5:], uldl_string[0], re.findall(r"\d+", tbs_string[0])[0]]
                    if '[DL]' in uldl_string:
                        csv_write_down.writerow(tbs_match)
                    else:
                        csv_write_up.writerow(tbs_match)


def time_difference(t1: str, t2: str) -> float:
    """
    Return the difference between two time points by seconds
    """
    t1_list = list(map(float, t1.split(':')))
    t2_list = list(map(float, t2.split(':')))
    time_diff = 3600 * (t2_list[0] - t1_list[0]) + 60 * (t2_list[1] - t1_list[1]) + ((t2_list[2]) - (t1_list[2]))
    return time_diff


def time_plus_one(time: str) -> str:
    """
    Add 1 second at current time point
    """
    time_list = list(map(str, time.split(':')))
    'Check carry-out of seconds'
    if time_list[2].split('.')[0] == '59':
        time_list[2] = '00.000000'
        'Check carry-out of minutes'
        if time_list[1] == '59':
            time_list[1] = '00'
            'Check carry-out of hours'
            if time_list[0] == '23':
                time_list[0] = '00'
            else:
                time_list[0] = str(int(time_list[0]) + 1)
                'Formatting 0X'
                if len(time_list[0]) == 1:
                    time_list[0] = '0' + time_list[0]
        else:
            time_list[1] = str(int(time_list[1]) + 1)
            'Formatting 0X'
            if len(time_list[1]) == 1:
                time_list[1] = '0' + time_list[1]
    else:
        temp = str(int(time_list[2].split('.')[0]) + 1)
        if len(temp) == 1:
            temp = '0' + temp
        time_list[2] = temp + '.000000'
    time_str = ':'.join(time_list)
    return time_str


def zero_complement(df):
    """
    Check time difference between adjacent time points
    if time difference is larger than 1 second
    insert a new row [time+=1, same rnti, same link, tbs=0]
    """
    for row_id in range(1, df.shape[0]):
        t1 = df.loc[row_id - 1, 'time']
        t2 = df.loc[row_id, 'time']
        if time_difference(t1, t2) > 1:
            new_row = df.loc[row_id - 1].copy()
            new_row['time'] = time_plus_one(new_row['time'])
            new_row['tbs'] = '0'
            df.loc[row_id - 0.5] = new_row
            df = df.sort_index().reset_index(drop=True)
    return df


def multiple_zero_complement(data_path: str) -> pd.DataFrame:
    """
    Due to the change of data size led by inserting zeros
    Multiple zero complements are required to complement
    the entire data set
    """
    df = pd.read_csv(data_path)
    while True:
        l1 = len(df)
        df = zero_complement(df)
        if len(df) == l1:
            return df


def find_time(df: pd.DataFrame, time: str) -> int:
    """
    find time point by the integer part
    lead to a small portion of redundant data in the data set
    """
    for index, row in df.iterrows():
        if row['time'][:8] == time[:8]:
            return index


def cut_off(df_DL, df_UL, log_path, start_id):
    """
    Cut off the data set by specified time periods
    When cut off, only check the integer part of time
    """
    df_cut_off = pd.DataFrame({'rnti': [], 'tbs_dl': [], 'tbs_ul': []})
    file = open(log_path)
    index = start_id
    for line in file:
        print(index)
        log = line.split(' ')
        time1 = log[1] + '.000000'
        # time2 = log[-1][:-1] + '.000000'
        time2 = log[-5] + '.000000'
        # print(time1,time2)
        id1 = find_time(df_DL, time1)
        id2 = find_time(df_DL, time2) + 1
        # print(time1,time2)
        id3 = find_time(df_UL, time1)
        id4 = find_time(df_UL, time2) + 1
        # print(id1,id2)
        """
        Cut off uplink and downlink tbs series from the row rbs series
        Generate the index(th) tbs series as '[time, rnti, tbs_dl, tbs_ul]' 
        """
        df_index_DL = df_DL.loc[df_DL.index[id2:id1], ['time', 'rnti', 'tbs_dl']]
        df_index_DL = df_index_DL.set_index('time')
        df_index_UL_tbs = df_UL.loc[df_UL.index[id4:id3], ['time', 'tbs_ul']]
        df_index_UL_rnti = df_UL.loc[df_UL.index[id4:id3], ['time', 'rnti']]
        df_index_UL_tbs = df_index_UL_tbs.set_index('time')
        df_index_UL_rnti = df_index_UL_rnti.set_index('time')
        df_index = pd.concat([df_index_DL, df_index_UL_rnti], axis=0)
        df_index = df_index.join(df_index_UL_tbs)
        df_index = df_index.fillna(0)
        df_index['series_id'] = np.ones(df_index.shape[0])*index
        df_cut_off = pd.concat([df_cut_off, df_index])
        index = index + 1
    return df_cut_off


def generate_traffic_files(dl_path, ul_path, RNTIlist):
    """
    :param dl_path: the file path to save downlink traffic block size (tbs_dl)
    :param ul_path: the file path to save uplink traffic block size (tbs_ul)
    :param RNTIlist: the list of rnti
    """
    f1 = open(dl_path, 'w', newline='')
    write_down = csv.writer(f1)
    write_down.writerow(['time', 'rnti', 'link', 'tbs_dl'])

    f2 = open(ul_path, 'w', newline='')
    write_up = csv.writer(f2)
    write_up.writerow(['time', 'rnti', 'link', 'tbs_ul'])

    for i in range(0, 30):
        print('read the %dth file' % i)
        try:
            if i == 0:
                path = "C:/Users/Jarvis/Desktop/Academic/Dissertation/Codes/2023.2.13_YOUTUBE/airscope.log"
                filter_data(path, write_down, write_up, RNTIlist)
            else:
                path = "C:/Users/Jarvis/Desktop/Academic/Dissertation/Codes/2023.2.13_YOUTUBE/airscope.log.%d" % i
                filter_data(path, write_down, write_up, RNTIlist)
        except:
            print("An exception occurred")
            break


def generate_traffic_zero_complement_file(dl_path, ul_path):
    df_new_down = multiple_zero_complement(dl_path)
    df_new_down.to_csv(dl_path, index=False)
    df_new_up = multiple_zero_complement(ul_path)
    df_new_up.to_csv(ul_path, index=False)


def generate_data_file(dl_path, ul_path, log_path, data_file_path, start_index):
    """
    :param dl_path: the file path to the downlink traffic block size (tbs_dl)
    :param ul_path: the file path to the uplink traffic block size (tbs_ul)
    :param log_path: the file path to the app usage log
    :param data_file_path: the file path to save the final tbs series
    :param start_index: the start index of the tbs series
    """
    df_new_down = pd.read_csv(dl_path)
    df_new_up = pd.read_csv(ul_path)
    df_cut_off_youtube = cut_off(df_new_down, df_new_up, log_path, start_index)
    df_cut_off_youtube.to_csv(data_file_path, index_label='time', mode='a')


if __name__ == "__main__":

    downlink_path = 'multidata_downlink.csv'
    uplink_path = 'multidata_uplink.csv'
    series_data_path = 'data_test.csv'
    app_log_path = 'C:/Users/Jarvis/Desktop/Academic/Dissertation/Codes/log files/log_YouTube.txt'
    RNTI_path = 'C:/Users/Jarvis/Desktop/Academic/Dissertation/Codes/log files/rnti_list.txt'
    id_str = 1
    RNTI_list = rnti_filter(RNTI_path)
    generate_traffic_files(downlink_path, uplink_path, RNTI_list)
    generate_traffic_zero_complement_file(downlink_path, uplink_path)
    generate_data_file(downlink_path, uplink_path, app_log_path, series_data_path, id_str)

    series_data_path = 'data_final.csv'
    data = pd.read_csv(series_data_path)
    for idd in range(201, 205):
        print(idd)
        dl_ul = data.loc[data.series_id == idd, ['tbs_dl', 'tbs_ul']]
        dl_ul.plot()
        plt.show()
    #
    # label = pd.DataFrame({'apps': ['ins', 'textchat', 'tiktok', 'textchat', 'spotify', 'videochat']*11})
    # label.to_csv(series_data_path, index=False)
