import xlrd
import math

label_path = r"D:\emnlp\video data\7\7.xlsx"

### get the all data for 3 classes ######################################################################################################
def excel_data(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols

    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list

label_list = excel_data(label_path)

list_0 = []
for i, j in enumerate(label_list):
    if j == 0:
        list_0.append(i)

list_1 = []
for i, j in enumerate(label_list):
    if j == 1:
        list_1.append(i)

print(len(list_0))
print(len(list_1))


from scipy.io import wavfile

def Energy(wave_data):
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
    return sum

for i in range(len(list_0)):
    All_E = 0
    audio_path = r"D:\emnlp\video data\7\Audio_Data\%d.wav"%(i+240)
    (sample_rate, sig) = wavfile.read(audio_path)
    wave_data = sig[:, 0]
    E = Energy(wave_data)
    All_E += E
    aver_e = All_E / len(list_0)

speech_energy_list = []
for j in range(len(list_1)):
    All_E = 0
    (sample_rate, sig) = wavfile.read(audio_path)
    wave_data = sig[:, 0]
    E = Energy(wave_data)
    each = E / aver_e
    speech_energy_list.append(each)
    x = 0
    x += each
    aver_speech = x / len(speech_energy_list)
    SNR = math.log10(aver_speech)

print(SNR)


