from sklearn.metrics import classification_report

list_dog = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
list_cat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
list_plane = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

test_label_list = list_dog + list_cat + list_plane
print(test_label_list)

list_dog = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2]
list_cat = [1, 1, 1, 1, 1, 2, 2, 2, 0, 0]
list_plane = [2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

final_pred_list = list_dog + list_cat + list_plane
print(final_pred_list)

print(classification_report(test_label_list, final_pred_list))