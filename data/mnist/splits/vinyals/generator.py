


if __name__ == '__main__':
    # class_names = []
    # with open('/Users/bytedance/Desktop/code/prototypical-networks/data/omniglot/splits/vinyals/trainval.txt', 'r') as f:
    #     for class_name in f.readlines():
    #         class_names.append(class_name.rstrip('\n')[:-7]) # 'Angelic/character01/rot000'
    # print(len(list(set(class_names))))
    # class_names = list(set(class_names))
    with open('/Users/waybaba/Desktop/code/prototypical-networks/data/mnist/splits/vinyals/odg_test.txt', 'w') as f:
        for class_name in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            for rot in ['000', '015', '030', '045', '060', '075']:
                line = 'num_' + str(class_name) + '/rot' + rot + '\n'
                f.write(line)
                print(line)

