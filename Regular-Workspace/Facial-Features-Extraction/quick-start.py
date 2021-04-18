import xlrd

if __name__ == '__main__':

    transcription_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t105 t060\clean_data\Copy of Dec4-2019 - t105 t060_cleaned_by_Kiana.xlsx"
    # prepare the segment time stamps according to the ground truth time stamps, read raw data from .xlsx file
    Speaker_List = []
    book = xlrd.open_workbook(transcription_path)
    sheet = book.sheet_by_index(0)

    for row_index in range(1, sheet.nrows):  # skip heading and 1st row
        time, speaker, text = sheet.row_values(row_index, end_colx=3)
        Speaker_List.append(speaker)

    for index in range(len(Speaker_List)-1):
        speaker_1 = Speaker_List[index]
        speaker_2 = Speaker_List[index+1]
        if(speaker_1 == speaker_2):
            print("ERROE! Repreated Speaker! Error Index is " + str(index+2))

    print("Speaker Information Checked!")