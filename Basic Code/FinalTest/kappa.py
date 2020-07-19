from sklearn.metrics import cohen_kappa_score
import xlrd

label_path_1 = r"C:\Users\Yingbo\Desktop\Kappa\With Jule\Jule.xlsx"
label_path_2 = r"C:\Users\Yingbo\Desktop\Kappa\With Jule\Yingbo.xlsx"

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

list1 = excel_data(label_path_1)
list2 = excel_data(label_path_2)

print("The cohen_kappa_score between two annotators is: ", cohen_kappa_score(list1, list2))

