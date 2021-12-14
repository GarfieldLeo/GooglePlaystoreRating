import pandas as pd
def makeCategory(filename):
    data = pd.read_csv(filename)['numerical']
    one_cat = []
    half_cat = []
    fifth_cat = []
    for i in data:
        # split into 1-2 2-3 3-4 4-5
        one_numr = int(i)
        if one_numr == 5:
            one_numr = 4
        one_cat.append(f"{one_numr}-{one_numr+1}")

        # split into 1.0-1.4 1.5-1.9 2.0-2.4 2.5-2.9....4.0-4.4 4.5-5
        half_whole = int(i)
        if half_whole == 5:
            half_cat.append("4.5-5.0")
        else:
            half_deci = (i - half_whole)*10
            half_deci = int(half_deci - half_deci % 5)
            half_cat.append(f"{half_whole}.{half_deci}-{half_whole}.{half_deci+4}")

        # split into 1.0-1.1 1.2-1.3 1.4-1.5 1.6-1.7 1.8-1.9 .....4.6-4.7 4.8-5.0
        fifth_whole = int(i)
        if fifth_whole == 5:
            fifth_cat.append("4.8-5.0")
        else:
            fifth_deci = 10 * (i - fifth_whole)
            fifth_deci = int(fifth_deci - fifth_deci % 2)
            fifth_cat.append(f"{fifth_whole}.{fifth_deci}-{fifth_whole}.{fifth_deci+1}")
    new_data = pd.DataFrame({"1":one_cat, "0.5":half_cat, "0.2":fifth_cat})
    new_data.to_csv(f"Cate{filename}")

makeCategory("yTest.csv")
makeCategory("yTrain.csv")