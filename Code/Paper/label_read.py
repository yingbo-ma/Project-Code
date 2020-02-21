import xlrd

label_path = r"D:\Data\Data_NC_State\TU405-6B\binary_label.xlsx"

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


print("Start reading Image & Label data...")
list = excel_data(label_path)

list_0 = []
for i, j in enumerate(list):
    if j == 0:
        list_0.append(i)

list_1 = []
for i, j in enumerate(list):
    if j == 1:
        list_1.append(i)

X_with_Class_0_Num = len(list_0)
X_with_Class_1_Num = len(list_1)

print(X_with_Class_0_Num)
print(X_with_Class_1_Num)

print(list_0)
print(list_1)