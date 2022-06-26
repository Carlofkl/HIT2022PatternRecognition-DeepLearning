import torch
import pandas as pd
from PIL import Image
import data_path as dp


def write_to_csv(net, trans, device, labels, csv_path):
    label_list = []
    name_list = []
    test_path = dp.get_test()

    for i in test_path:
        net.eval()
        image = Image.open(i).convert('RGB')
        img = trans(image)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)
        output = net(img)
        name = i.split('/')[1]
        _, prediction = torch.max(output.data, dim=1)
        label = labels[prediction]
        name_list.append(name)
        label_list.append(label)

    finally_result = pd.DataFrame({'file': name_list, 'species': label_list})
    print("writing to " + csv_path + '...')
    finally_result.to_csv(csv_path, index=False)
