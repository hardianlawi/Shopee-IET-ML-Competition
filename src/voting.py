import numpy
import csv
import os
import glob

category_count = 18
test_rows = 16111
OUTPUT_FILE = '../outputs/submissions/result.csv'

# file_paths = []
# for model_type in ["DenseNet121", "DenseNet169", "DenseNet201", "InceptionResNetV2", "InceptionV3", "ResNet50", "Xception"]:
#     for path in glob.glob(os.path.join("../outputs/test/", model_type, "*")):
#         file_paths.append(path)
#         print(path)

test_outputs = {
    'ResNet50': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/ResNet50/ResNet50_test_iter1.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter2.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter3.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter4.csv",
            "../outputs/test/ResNet50/ResNet50_test_iter5.csv",
        ],
    },
    'InceptionResNetV2': {
        'type': 'TEST_ITER',
        'weight': 1.5,
        'file_paths': [
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter1.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter6.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter3.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter0.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter2.csv",
            "../outputs/test/InceptionResNetV2/InceptionResNetV2_test_iter5.csv",
        ],
    },
    'InceptionV3': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/InceptionV3/InceptionV3_test_iter0.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter5.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter6.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter1.csv",
            "../outputs/test/InceptionV3/InceptionV3_test_iter3.csv",
        ],
    },
    'DenseNet201': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/DenseNet201/DenseNet201_test_iter5.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter6.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter2.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter0.csv",
            "../outputs/test/DenseNet201/DenseNet201_test_iter4.csv",
        ],
    },
    'DenseNet121': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/DenseNet121/DenseNet121_test_iter2.csv",
            "../outputs/test/DenseNet121/DenseNet121_test_iter3.csv",
        ],
    },
    'DenseNet169': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/DenseNet169/DenseNet169_test_iter2.csv",
            "../outputs/test/DenseNet169/DenseNet169_test_iter1.csv",
            "../outputs/test/DenseNet169/DenseNet169_test_iter5.csv",
            "../outputs/test/DenseNet169/DenseNet169_test_iter4.csv",
        ],
    },
    'Xception': {
        'type': 'TEST_ITER',
        'weight': 1.5,
        'file_paths': [
            "../outputs/test/Xception/Xception_test_iter0.csv",
            "../outputs/test/Xception/Xception_test_iter1.csv",
            "../outputs/test/Xception/Xception_test_iter2.csv",
            "../outputs/test/Xception/Xception_test_iter3.csv",
            "../outputs/test/Xception/Xception_test_iter4.csv",
            "../outputs/test/Xception/Xception_test_iter5.csv",
            "../outputs/test/Xception/Xception_test_iter6.csv",
        ],
    },
    'NASNetLarge': {
        'type': 'TEST_ITER',
        'weight': 1,
        'file_paths': [
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter1.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter2.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter3.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter0.csv",
            "../outputs/test/NASNetLarge/NASNetLarge_test_iter4.csv",
        ],
    },

    # 'resnet3': {
    #     'type': 'TEST_RESULT',
    #     'weight': 1,
    #     'file_path': 'voting28.csv'
    # },
}


def voting_iter(rows, maps, weight=1):
    for row_num, row in enumerate(rows):
        ans = numpy.array([float(x) for x in row]).argmax(axis=0)
        maps[row_num][ans] += 1.0 * weight


def voting_result(rows, maps, weight=1):
    for row_num, row in enumerate(rows):
        maps[row_num][row[1]] += 1.0 * weight


answer_maps = [[0] * category_count for _ in range(test_rows)]

for key, value in test_outputs.items():

    output_type = value['type']

    if output_type == 'TEST_ITER':
        weight, file_paths = float(value['weight']), value['file_paths']
        single_answer_maps = [[0] * category_count for _ in range(test_rows)]
        for file_path in file_paths:
            file = open(file_path)
            reader = csv.reader(file)
            rows = [row[1:] for row in reader][1:]
            voting_iter(rows, single_answer_maps)
        voting_iter(single_answer_maps, answer_maps, weight)

    elif output_type == 'TEST_RESULT':
        weight, file_path = float(value['weight']), value['file_path']
        file = open(file_path)
        reader = csv.reader(file)
        rows = [row for row in reader][1:]
        rows = [map(int, row) for row in rows]
        voting_result(rows, answer_maps, weight)

result = []
for row_num, row in enumerate(answer_maps):
    row = [float(x) for x in row]
    ans = numpy.array(row).argmax(axis=0)
    maxs = row[ans]
    same_ans = [idx for idx, x in enumerate(row) if x == maxs]
    if len(same_ans) > 1:
        print('answer for question number {} has {} candidates: {}, choosing {}'.format(row_num + 1, len(same_ans), same_ans, ans))
    result.append([row_num + 1, ans])

file = open(OUTPUT_FILE, 'w')
writer = csv.writer(file)
writer.writerows([['id', 'category']] + result)
