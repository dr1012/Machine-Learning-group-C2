from tabulate import tabulate

def drawTable(headers,data):

    # headers = ["","LR","Polynomial","RBF","kNN"]
    # headers2 = ["Polynomial Regression","Radial Basis Function","kNN"]

    # data = [["RMSE","0.654", "0.64", "0.635","0.814"],
    #         ["STE", "+-" + str(float('%.3g' % 0.001)), float('%.3g' % 0.002), float('%.3g' % 0.002),
    #          float('%.3g' % 0.003)]]

    # table2 = [["subset: " + str("0, 9, 1, 2, 7, 6"),"reg param = " + str("3.9810717055349694e-10"),"param-s: " + str("[10.0], 0.696993777439047, 54.0")],
    #           ["degree: " + str("2.0"), "scale " + str("206.913808111479"),"subset: " + str("10.0")],
    #           [" ","centres: " + str("0.05"),"k = " + str("54.0")]]

    print(tabulate(data,headers=headers,tablefmt="grid"))
    # print(tabulate(table2,headers=headers2,tablefmt="grid"))


# drawTable()